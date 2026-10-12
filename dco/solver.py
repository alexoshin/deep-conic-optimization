"""COSMO implementation."""

import time
import types
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from jaxtyping import Float, Int
import torch
from torch import Tensor, nn

from .cg import identity_preconditioner, jacobi_preconditioner
from .cones import BATCHED_CONE_TYPES, Cone, build_super_cones
from .functional import (
    Solver_Residuals,
    admm_residuals,
    clamp_rhos,
    converged_flags,
    dual_update,
    kkt_rhs,
    recover_z_tilde,
    reduced_rhs,
    rho_change_gate,
    second_block,
)
from .linalg import lu_factor, lu_solve
from .linsys import AdjointWarmStartCache, ReducedSystemOperator, solve_reduced_system
from .policy import HyperparameterPolicy, PolicyState
from .problem import SuperProblem
from .super_batch import BatchMetadata
from .utils import format_time
from .warmstart import Initialization

# CG tolerances for the adjoint solve of the indirect method
DEFAULT_CG_BACKWARD_EPS_ABS = 0.0
DEFAULT_CG_BACKWARD_EPS_REL = 1e-6

LINSYS_METHODS = ("direct", "indirect")


class Solver_Status(Enum):
    NOT_SOLVED = 0
    CONVERGED = 1
    MAX_ITER_REACHED = 2
    TIME_LIMIT_REACHED = 3
    PRIMAL_INFEASIBLE = 4
    DUAL_INFEASIBLE = 5


@dataclass
class Solver_Result:
    """Result for one solve.

    When `save_history` is `False` the `*_hist` iterate tensors hold only the final entry
    (the scalar residual histories always contain the full history).
    """

    x_hist: Float[Tensor, "... K+1 n"]
    z_hist: Float[Tensor, "... K+1 m"]
    y_hist: Float[Tensor, "... K+1 m"]
    rhos_hist: Float[Tensor, "... K+1 m"]
    alpha_hist: Float[Tensor, "... K+1"]
    obj_val_hist: Float[Tensor, "... K+1"]
    dual_res_norm_hist: Float[Tensor, "... K+1"]
    prim_res_norm_hist: Float[Tensor, "... K+1"]
    status: Solver_Status
    solve_time: float
    setup_time: float
    coeff_fac_iters: list[int]
    cg_iters: list[int]

    @property
    def converged(self) -> bool:
        return self.status == Solver_Status.CONVERGED

    @property
    def x(self) -> Float[Tensor, "... n"]:
        return self.x_hist[..., -1, :]

    @property
    def z(self) -> Float[Tensor, "... m"]:
        """The slack `b - A x` at optimality."""
        return self.z_hist[..., -1, :]

    @property
    def y(self) -> Float[Tensor, "... m"]:
        return self.y_hist[..., -1, :]

    @property
    def num_iters(self) -> int:
        return max(self.x_hist.shape[-2], self.prim_res_norm_hist.shape[-1]) - 1

    @property
    def mean_cg_iters(self) -> float:
        return float(np.mean(self.cg_iters)) if self.cg_iters else 0.0


class ProjectionProx(nn.Module):
    """The second ADMM block: Euclidean projection onto `K`."""

    def __init__(
        self,
        cones: list[Cone] | list[list[Cone]],
        batch_metadata: BatchMetadata | None = None,
        fuse_cones: bool = True,
    ) -> None:
        """Build the projection.

        Args:
            cones: One cone list, or one list per sub-problem of a super-batch
                (a single list under `batch_metadata` is replicated across the batch).
            batch_metadata: Super-batch layout, or None for a single problem.
                A dense `(B, m, n)` batch is solved as one uniform super-batch,
                so its projection must be built with `dco.problem.uniform_batch_metadata(B, n, m)`.
            fuse_cones: Try to group same-type cones of a single problem together so each type is one batched op
                (only applied when it reduces the number of passes; always on for super-batched).
        """
        super().__init__()
        if batch_metadata is not None:
            cones_per_problem = [cones] * batch_metadata.B if cones and isinstance(cones[0], Cone) else cones
            self.cones = build_super_cones(
                cones_per_problem, m_offsets=batch_metadata.m_offsets, soc_strategy=batch_metadata.soc_strategy
            )
        else:
            self.cones = self.maybe_fuse_cones(list(cones), fuse_cones)

    @staticmethod
    def maybe_fuse_cones(cones: list[Cone], fuse_cones: bool) -> list[Cone]:
        if not fuse_cones or not cones or any(isinstance(cone, BATCHED_CONE_TYPES) for cone in cones):
            return cones
        fused = build_super_cones([cones], m_offsets=[0], soc_strategy="auto")
        return cones if len(fused) >= len(cones) else fused

    def forward(self, v: Float[Tensor, "... m"]) -> Float[Tensor, "... m"]:
        return self.project(v)

    def project(self, v: Float[Tensor, "... m"]) -> Float[Tensor, "... m"]:
        """Compute `Pi_K(v)`."""
        for cone in self.cones:
            v = cone(v)
        return v

    def violation(self, v: Float[Tensor, "... m"]) -> Float[Tensor, "... m"]:
        """Compute `v - Pi_K(v)`."""
        return v - self.project(v)

    @property
    def blocks(self) -> list[Cone]:
        return self.cones


class Solver:
    """COSMO solver."""

    def __init__(
        self,
        P: Float[Tensor, "... n n"],
        q: Float[Tensor, "... n"],
        A: Float[Tensor, "... m n"],
        b: Float[Tensor, "... m"],
        prox: ProjectionProx,
        policy: HyperparameterPolicy | None = None,
        initialization: Initialization | None = None,
        sigma: float = 1e-6,
        batch_metadata: BatchMetadata | None = None,
        structure_dispatch: bool = True,
    ) -> None:
        """Initialize the solver.

        Args:
            P, q, A, b: Problem data.
            prox: The projection onto `K`.
            policy: Hyperparameter policy (default: fixed rho and alpha).
            initialization: Warm start (default: zeros).
            sigma: Proximal regularization keeping `P + sigma I` positive definite.
            batch_metadata: Layout of block-diagonal data, or None.
            structure_dispatch: Skip the `P` product when `P` is exactly zero or diagonal,
                and adopt new rhos on the device instead of a host branch in indirect mode.
        """
        self.prox = prox
        self.policy = policy if policy is not None else HyperparameterPolicy()
        self.initialization = initialization if initialization is not None else Initialization()
        self.sigma = sigma
        self.structure_dispatch = structure_dispatch

        self.problem = SuperProblem.build(P, q, A, b, batch_metadata, structure_dispatch)
        self.batch_metadata = self.problem.metadata
        self.P = self.problem.P_matrix()
        self.q, self.A, self.b = self.problem.q, self.problem.A, self.problem.b
        self.m, self.n = self.problem.m, self.problem.n
        self.tensor_kwargs = {"dtype": self.q.dtype, "device": self.q.device}

        covered = sum(int(block.indices.numel()) for block in prox.blocks)
        if covered != self.m:
            raise ValueError(
                f"The projection covers {covered} constraint rows, but the problem has m={self.m}. "
                f"A dense (B, m, n) batch is solved as one uniform super-batch, "
                f"so its projection must be built with batch_metadata=uniform_batch_metadata(B, n, m)."
            )
        for block in prox.blocks:
            block.to(device=self.q.device)

    def solve(
        self,
        max_iter: int = 1000,
        eps_abs: float = 1e-3,
        eps_rel: float = 0.0,
        eps_prim_inf: float = 1e-4,
        eps_dual_inf: float = 1e-4,
        time_limit: float | None = None,
        verbose: bool = False,
        print_every: int = 25,
        check_termination: int = 25,
        linsys_solve_method: str = "direct",
        adaptive_rho_tolerance: float = 5.0,
        cg_max_iter: int | None = None,
        cg_eps_abs: float = 1e-5,
        cg_eps_rel: float = 0.0,
        cg_verbose: bool = False,
        cg_preconditioner: str = "none",
        cg_check_every: int = 1,
        cg_backward_eps_abs: float = DEFAULT_CG_BACKWARD_EPS_ABS,
        cg_backward_eps_rel: float = DEFAULT_CG_BACKWARD_EPS_REL,
        cg_backward_max_iter: int | None = None,
        adjoint_warm_start: bool = True,
        save_history: bool = True,
        callbacks: Iterable[Callable[[types.SimpleNamespace], None]] = (),
    ) -> Solver_Result:
        """Solve the conic program.

        Args:
            max_iter: Max number of iterations to run.
            eps_abs, eps_rel: Convergence tolerances on the ADMM residuals.
            eps_prim_inf, eps_dual_inf: Infeasibility-certificate tolerances; 0 disables the check.
            time_limit: Wall-clock time limit, in seconds.
            verbose, print_every: Progress printing.
            check_termination: Number of iterations between convergence and infeasibility checks.
                `<= 0` disables the check and runs exactly `max_iter` iterations (e.g. for a training unroll).
            linsys_solve_method:
                - `direct`: factorizes the dense `(n+m) x (n+m)` KKT matrix and refactorizes it whenever rho changes
                - `indirect`: solves the reduced normal equations form using CG with the matrix-free implicit backward
            adaptive_rho_tolerance: Factor determing how large of a change in rho is necessary before adopting it.
                `1.0` means the new rho is always adopted.
            cg_*: CG settings for the forward pass (indirect method only).
                - `cg_preconditioner`: `"none"` or `"jacobi"` (the diagonal of the matrix).
            cg_backward_*: CG settings for the backward pass adjoint solve (`None` uses the forward setting).
            adjoint_warm_start: Warm-start each iteration's adjoint solve from the previous iteration's.
            save_history: Keep every iterate (needed by the training loss). `False` keeps only the last iterate.
            callbacks: Called after each iteration under `no_grad` with that iteration's state.
        """
        if linsys_solve_method not in LINSYS_METHODS:
            raise ValueError(f"linsys_solve_method must be one of {LINSYS_METHODS}, got {linsys_solve_method!r}")
        if cg_preconditioner not in ("none", "jacobi"):
            raise ValueError(f"cg_preconditioner must be 'none' or 'jacobi', got {cg_preconditioner!r}")
        use_indirect = linsys_solve_method == "indirect"
        if self.problem.B > 1 and not use_indirect:
            warnings.warn(
                "The direct method on a batch densifies and factorizes the whole super KKT matrix; "
                "use 'indirect' for memory-efficient batched solves.",
                stacklevel=2,
            )

        setup_start = time.time()
        x, z, y = self.initialization(
            self.P, self.q, self.A, self.b, self.prox.blocks, batch_metadata=self.batch_metadata
        )
        x_tilde, z_tilde, x_prev, z_prev = x.clone(), z.clone(), x.clone(), z.clone()
        residuals = self.compute_residuals(x, z, y, x_tilde, z_tilde, x_prev, z_prev)
        policy_state: PolicyState = self.policy.initialize(
            self.P, self.q, self.A, self.b, self.prox.blocks, batch_metadata=self.batch_metadata
        )
        setup_time = time.time() - setup_start

        x_hist, z_hist, y_hist = [x], [z], [y]
        rhos_hist: list[Tensor] = []
        alpha_hist: list[Tensor] = []
        obj_val_hist = [residuals.obj_val]
        dual_res_norm_hist = [self.dual_res_norm(residuals.dual_res)]
        prim_res_norm_hist = [self.prim_res_norm(residuals.prim_res)]
        coeff_fac_iters: list[int] = []
        cg_iters: list[int] = []
        status = Solver_Status.NOT_SOLVED

        solve_start = time.time()
        rhos, alpha, policy_state = self.policy(x, z, y, residuals, 0, policy_state)
        rhos = clamp_rhos(rhos)
        rhos_hist.append(rhos)
        alpha_hist.append(self.per_problem(alpha))

        # First-block setup: compute the KKT factorization or the matrix-free operator of the reduced system
        coeff = coeff_fac = linop = None
        adjoint_cache = AdjointWarmStartCache() if (use_indirect and adjoint_warm_start) else None
        if use_indirect:
            linop = ReducedSystemOperator(
                self.P,
                self.A,
                rhos,
                sigma=self.sigma,
                structure_dispatch=self.structure_dispatch,
                A_T=self.problem.A_T,
                A_sq_T=self.problem.A_sq_T,
            )
        else:
            coeff = self.construct_coefficient_matrix(rhos)
            coeff_fac = self.factorize(coeff, 0)
            coeff_fac_iters.append(0)
        blend_rhos = use_indirect and self.structure_dispatch

        if verbose:
            self.print_header(max_iter, eps_abs, eps_rel, eps_prim_inf, eps_dual_inf, linsys_solve_method)
            print(get_result_str(0, obj_val_hist[-1], prim_res_norm_hist[-1], dual_res_norm_hist[-1], rhos))

        # Run ADMM iterations
        k = -1
        for k in range(max_iter):
            # First block: solve the KKT system
            if use_indirect:
                x_tilde, z_tilde, num_cg_iters = self.first_block_update_indirect(
                    linop,
                    x,
                    z,
                    y,
                    rhos,
                    x_tilde,
                    cg_max_iter,
                    cg_eps_abs,
                    cg_eps_rel,
                    cg_verbose,
                    cg_preconditioner,
                    cg_backward_eps_abs,
                    cg_backward_eps_rel,
                    cg_backward_max_iter,
                    adjoint_cache,
                    cg_check_every,
                )
                cg_iters.append(num_cg_iters)
            else:
                x_tilde, z_tilde = self.first_block_update(coeff_fac, x, z, y, rhos)

            # Second block: projection
            x_prev, z_prev = x.clone(), z.clone()
            x, z, z_relaxed = self.second_block_update(x_tilde, z_tilde, x, z, y, rhos, alpha)
            if save_history:
                x_hist.append(x)
                z_hist.append(z)

            # Dual update
            y_prev = y.clone()
            y = dual_update(y, rhos, z_relaxed, z)
            if save_history:
                y_hist.append(y)

            residuals = self.compute_residuals(x, z, y, x_tilde, z_tilde, x_prev, z_prev)
            obj_val_hist.append(residuals.obj_val)
            dual_res_norm = self.dual_res_norm(residuals.dual_res)
            prim_res_norm = self.prim_res_norm(residuals.prim_res)
            dual_res_norm_hist.append(dual_res_norm)
            prim_res_norm_hist.append(prim_res_norm)

            if callbacks:
                with torch.no_grad():
                    frame = types.SimpleNamespace(
                        k=k,
                        x=x,
                        z=z,
                        y=y,
                        x_tilde=x_tilde,
                        z_tilde=z_tilde,
                        rhos=rhos,
                        alpha=alpha,
                        residuals=residuals,
                        prox=self.prox,
                    )
                    for callback in callbacks:
                        callback(frame)

            if verbose and print_every > 0 and (k + 1) % print_every == 0:
                print(get_result_str(k + 1, residuals.obj_val, prim_res_norm, dual_res_norm, rhos))

            if check_termination > 0 and k % check_termination == 0:
                converged = converged_flags(
                    dual_res_norm, prim_res_norm, residuals.dual_rel_norm, residuals.prim_rel_norm, eps_abs, eps_rel
                )
                if torch.all(converged):
                    status = Solver_Status.CONVERGED
                    break
                if eps_prim_inf > 0.0 and self.check_primal_infeasible(y, y_prev, eps_prim_inf):
                    status = Solver_Status.PRIMAL_INFEASIBLE
                    break
                if eps_dual_inf > 0.0 and self.check_dual_infeasible(x, x_prev, eps_dual_inf):
                    status = Solver_Status.DUAL_INFEASIBLE
                    break
                if time_limit is not None and (time.time() - solve_start) >= time_limit:
                    status = Solver_Status.TIME_LIMIT_REACHED
                    break

            # Compute new hyperparameters
            rhos_new, alpha, policy_state_new = self.policy(x, z, y, residuals, k + 1, policy_state)
            rhos_new = clamp_rhos(rhos_new)
            if blend_rhos:
                fire = rho_change_gate(rhos, rhos_new, adaptive_rho_tolerance)
                rhos = torch.where(fire, rhos_new, rhos)
                policy_state = PolicyState(
                    rhos=torch.where(fire, policy_state_new.rhos, policy_state.rhos),
                    alpha=torch.where(fire, policy_state_new.alpha, policy_state.alpha),
                )
            elif rho_change_gate(rhos, rhos_new, adaptive_rho_tolerance):
                rhos, policy_state = rhos_new, policy_state_new
                if not use_indirect:
                    coeff = self.update_coefficient_matrix(coeff, rhos)
                    coeff_fac = self.factorize(coeff, k)
                    coeff_fac_iters.append(k)

            if save_history:
                rhos_hist.append(rhos)
                alpha_hist.append(self.per_problem(alpha))

        solve_time = time.time() - solve_start
        if status == Solver_Status.NOT_SOLVED:
            status = Solver_Status.MAX_ITER_REACHED
        if verbose:
            self.print_summary(
                status, k + 1, obj_val_hist[-1], prim_res_norm_hist[-1], dual_res_norm_hist[-1], setup_time, solve_time
            )
        if not save_history:
            x_hist, z_hist, y_hist, rhos_hist, alpha_hist = [x], [z], [y], [rhos], [self.per_problem(alpha)]

        return Solver_Result(
            x_hist=torch.stack(x_hist, dim=-2),
            z_hist=torch.stack(z_hist, dim=-2),
            # Negated so the returned dual is the KKT multiplier y^* in K^*
            y_hist=-torch.stack(y_hist, dim=-2),
            rhos_hist=torch.stack(rhos_hist, dim=-2),
            alpha_hist=torch.stack(alpha_hist, dim=-1),
            obj_val_hist=torch.stack(obj_val_hist, dim=-1),
            dual_res_norm_hist=torch.stack(dual_res_norm_hist, dim=-1),
            prim_res_norm_hist=torch.stack(prim_res_norm_hist, dim=-1),
            status=status,
            solve_time=solve_time,
            setup_time=setup_time,
            coeff_fac_iters=coeff_fac_iters,
            cg_iters=cg_iters,
        )

    def matvec_A(self, v: Float[Tensor, "... n"]) -> Float[Tensor, "... m"]:
        return self.problem.matvec_A(v)

    def matvec_AT(self, v: Float[Tensor, "... m"]) -> Float[Tensor, "... n"]:
        return self.problem.matvec_AT(v)

    def matvec_P(self, v: Float[Tensor, "... n"]) -> Float[Tensor, "... n"]:
        return self.problem.matvec_P(v)

    def per_problem(self, v: Float[Tensor, "... B"]) -> Float[Tensor, "..."]:
        """If the original problem was a single problem, drop the leading `B = 1` dimension."""
        return v.squeeze(-1) if self.problem.source == "single" else v

    def dual_res_norm(self, dual_res: Float[Tensor, "... n"]) -> Float[Tensor, "..."]:
        reducer_n, _ = self.problem.reducers(dual_res.device)
        return self.per_problem(reducer_n.inf_norm(dual_res))

    def prim_res_norm(self, prim_res: Float[Tensor, "... m"]) -> Float[Tensor, "..."]:
        _, reducer_m = self.problem.reducers(prim_res.device)
        return self.per_problem(reducer_m.inf_norm(prim_res))

    def broadcast_alpha(self, alpha: Float[Tensor, "..."], space: str) -> Float[Tensor, "..."]:
        """Broadcast alpha over each sub-problem's variables (`"n"`) or rows (`"m"`)."""
        if alpha.dim() == 0:
            return alpha
        reducer_n, reducer_m = self.problem.reducers(alpha.device)
        return (reducer_n if space == "n" else reducer_m).broadcast(alpha)

    def reductions(self, device: torch.device) -> tuple[Callable, Callable, Callable]:
        reducer_n, reducer_m = self.problem.reducers(device)
        return (
            lambda a, c: self.per_problem(reducer_n.dot(a, c)),
            lambda *vs: self.per_problem(reducer_n.max_inf_norm(*vs)),
            lambda *vs: self.per_problem(reducer_m.max_inf_norm(*vs)),
        )

    def construct_coefficient_matrix(self, rhos: Float[Tensor, "... m"]) -> Float[Tensor, "n+m n+m"]:
        """Compute the dense KKT matrix `[[P + sigma I, A^T], [A, -diag(1/rho)]]`."""
        coeff = torch.zeros(self.n + self.m, self.n + self.m, **self.tensor_kwargs)
        A_dense = self.problem.A_dense()
        coeff[: self.n, : self.n] = self.problem.P_dense() + self.sigma * torch.eye(self.n, **self.tensor_kwargs)
        coeff[: self.n, self.n :] = A_dense.mT
        coeff[self.n :, : self.n] = A_dense
        coeff[self.n :, self.n :] = -torch.diag_embed(1.0 / rhos)
        return coeff

    def update_coefficient_matrix(self, coeff: Tensor, rhos: Float[Tensor, "... m"]) -> Tensor:
        """Update the KKT matrix with new `rhos`."""
        coeff_new = coeff.clone()
        coeff_new[self.n :, self.n :] = -torch.diag_embed(1.0 / rhos)
        return coeff_new

    @staticmethod
    def factorize(coeff: Tensor, k: int) -> tuple[Tensor, Int[Tensor, "..."]]:
        """Factor the KKT matrix."""
        coeff_fac, successes = lu_factor(coeff)
        if not torch.all(successes):
            raise ValueError(f"The KKT matrix is singular at iteration {k}")
        return coeff_fac

    def first_block_update(
        self,
        coeff_fac: tuple[Tensor, Tensor],
        x: Float[Tensor, "... n"],
        z: Float[Tensor, "... m"],
        y: Float[Tensor, "... m"],
        rhos: Float[Tensor, "... m"],
    ) -> tuple[Float[Tensor, "... n"], Float[Tensor, "... m"]]:
        """Solve the KKT system using its LU factors and return `(x_tilde, z_tilde)`."""
        rhs = kkt_rhs(self.sigma, x, self.q, self.b, z, y, rhos)
        xnu = lu_solve(coeff_fac, rhs.unsqueeze(-1)).squeeze(-1)
        return xnu[..., : self.n], recover_z_tilde(z, xnu[..., self.n :], y, rhos)

    def first_block_update_indirect(
        self,
        linop: ReducedSystemOperator,
        x: Float[Tensor, "... n"],
        z: Float[Tensor, "... m"],
        y: Float[Tensor, "... m"],
        rhos: Float[Tensor, "... m"],
        x_tilde: Float[Tensor, "... n"],
        cg_max_iter: int | None,
        cg_eps_abs: float,
        cg_eps_rel: float,
        cg_verbose: bool,
        cg_preconditioner: str,
        cg_backward_eps_abs: float | None,
        cg_backward_eps_rel: float | None,
        cg_backward_max_iter: int | None,
        adjoint_cache: AdjointWarmStartCache | None,
        cg_check_every: int,
    ) -> tuple[Float[Tensor, "... n"], Float[Tensor, "... m"], int]:
        """Solve the reduced KKT system using CG and return `(x_tilde, z_tilde)`."""
        rhs = reduced_rhs(self.sigma, x, self.q, self.b, z, y, rhos, self.matvec_AT)
        if cg_preconditioner == "jacobi":
            with torch.no_grad():
                linop.update_rhos(rhos)
                preconditioner = jacobi_preconditioner(linop.diagonal())
        else:
            preconditioner = identity_preconditioner
        x_tilde, num_cg_iters = solve_reduced_system(
            linop,
            rhos,
            rhs,
            x0=x_tilde,
            preconditioner=preconditioner,
            max_iter=cg_max_iter,
            eps_abs=cg_eps_abs,
            eps_rel=cg_eps_rel,
            backward_eps_abs=cg_backward_eps_abs,
            backward_eps_rel=cg_backward_eps_rel,
            backward_max_iter=cg_backward_max_iter,
            adjoint_cache=adjoint_cache,
            check_every=cg_check_every,
        )
        if cg_verbose:
            print(f"CG: {num_cg_iters} iterations")
        return x_tilde, self.b - self.matvec_A(x_tilde), num_cg_iters

    def second_block_update(
        self,
        x_tilde: Float[Tensor, "... n"],
        z_tilde: Float[Tensor, "... m"],
        x: Float[Tensor, "... n"],
        z: Float[Tensor, "... m"],
        y: Float[Tensor, "... m"],
        rhos: Float[Tensor, "... m"],
        alpha: Float[Tensor, "..."],
    ) -> tuple[Float[Tensor, "... n"], Float[Tensor, "... m"], Float[Tensor, "... m"]]:
        """Compute the second ADMM block update."""
        return second_block(
            self.prox.project,
            x_tilde,
            z_tilde,
            x,
            z,
            y,
            rhos,
            self.broadcast_alpha(alpha, "n"),
            self.broadcast_alpha(alpha, "m"),
        )

    def compute_residuals(self, x, z, y, x_tilde, z_tilde, x_prev, z_prev) -> Solver_Residuals:
        dot_n, max_inf_norm_n, max_inf_norm_m = self.reductions(x.device)
        return admm_residuals(
            self.q,
            self.b,
            x,
            z,
            x_tilde,
            z_tilde,
            x_prev,
            z_prev,
            self.matvec_P(x),
            self.matvec_A(x),
            self.matvec_AT(y),
            dot_n,
            max_inf_norm_n,
            max_inf_norm_m,
        )

    def check_primal_infeasible(self, y: Tensor, y_prev: Tensor, eps_prim_inf: float) -> bool:
        reducer_n, reducer_m = self.problem.reducers(y.device)
        delta_y = y - y_prev
        delta_y_norm = reducer_m.inf_norm(delta_y)
        if not torch.all(delta_y_norm > 0):
            return False
        tol = eps_prim_inf * delta_y_norm
        if not torch.all(reducer_n.inf_norm(self.matvec_AT(delta_y)) <= tol):
            return False
        if not torch.all(reducer_m.inf_norm(self.prox.project(delta_y)) <= tol):
            return False
        return bool(torch.all(reducer_m.dot(self.b, delta_y) >= tol))

    def check_dual_infeasible(self, x: Tensor, x_prev: Tensor, eps_dual_inf: float) -> bool:
        reducer_n, reducer_m = self.problem.reducers(x.device)
        delta_x = x - x_prev
        delta_x_norm = reducer_n.inf_norm(delta_x)
        if not torch.all(delta_x_norm > 0):
            return False
        tol = eps_dual_inf * delta_x_norm
        if not torch.all(reducer_n.inf_norm(self.matvec_P(delta_x)) <= tol):
            return False
        if not torch.all(reducer_n.dot(self.q, delta_x) <= tol):
            return False
        neg_A_delta_x = -self.matvec_A(delta_x)
        return bool(torch.all(reducer_m.inf_norm(self.prox.violation(neg_A_delta_x)) <= tol))

    def print_header(self, max_iter, eps_abs, eps_rel, eps_prim_inf, eps_dual_inf, linsys_solve_method) -> None:
        print("-" * 34 + " SOLVER " + "-" * 34)
        print(f"  sub-problems: {self.problem.B} ({self.problem.source}), n: {self.n}, m: {self.m}")
        print(f"  blocks: {[block.__class__.__name__ for block in self.prox.blocks[:5]]}")
        print(f"  policy: {self.policy.__class__.__name__}, initialization: {self.initialization.__class__.__name__}")
        print(f"  max iter = {max_iter}, sigma = {self.sigma:.1e}, eps abs = {eps_abs:.1e}, eps rel = {eps_rel:.1e}")
        print(f"  eps prim inf = {eps_prim_inf:.1e}, eps dual inf = {eps_dual_inf:.1e}, linear solve = {linsys_solve_method}")  # fmt: skip
        print("-" * 76)

    @staticmethod
    def print_summary(status, num_iters, obj_vals, prim_res_norms, dual_res_norms, setup_time, solve_time) -> None:
        print("-" * 76)
        print(f"Result: {status.name.lower().replace('_', ' ')}")
        print(f"Final objective value:      {obj_vals.max().item(): .4e}")
        print(f"Final primal residual norm: {prim_res_norms.max().item(): .4e}")
        print(f"Final dual residual norm:   {dual_res_norms.max().item(): .4e}")
        print(f"Setup time: {format_time(setup_time)}, solve time: {format_time(solve_time)}, iterations: {num_iters}")
        print("-" * 76 + "\n")


def get_result_str(k: int, obj_vals: Tensor, prim_res_norms: Tensor, dual_res_norms: Tensor, rhos: Tensor) -> str:
    header = " iteration   objective   prim res    dual res    rho min     rho max \n" if k == 0 else ""
    return header + (
        f"{k:^10d}  {obj_vals.max().item():>10.3e}  {prim_res_norms.max().item():>10.3e}  "
        f"{dual_res_norms.max().item():>10.3e}  {rhos.min().item():>10.3e}  {rhos.max().item():>10.3e}"
    )
