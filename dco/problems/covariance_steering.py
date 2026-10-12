"""Covariance-steering SCP problems."""

from collections.abc import Iterator
from dataclasses import dataclass
import time
from typing import Protocol

from jaxtyping import Float
import torch
from torch import Tensor

from ..clarabel import ClarabelSCPSolver, make_solver_result, solve as solve_with_clarabel
from ..cones import Cone, NonnegativeOrthant, PositiveSemidefiniteCone, SecondOrderCone, ZeroCone
from ..policy import AdaptiveRhoOSQP, HyperparameterPolicy
from ..solver import ProjectionProx, Solver, Solver_Result
from ..equilibration import equilibriate_
from ..linalg import svec


def double_integrator_dynamics(
    dt: float,
    sigma_a: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Float[Tensor, "4 4"], Float[Tensor, "4 2"], Float[Tensor, "4 4"]]:
    """Exact ZOH discretization of the 2D double integrator with diagonal acceleration noise."""
    A = torch.tensor(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    B = torch.tensor(
        [
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=dtype,
        device=device,
    )
    W = torch.diag(
        torch.tensor(
            [
                (dt**3) * sigma_a**2 / 3.0,
                (dt**3) * sigma_a**2 / 3.0,
                dt * sigma_a**2,
                dt * sigma_a**2,
            ],
            dtype=dtype,
            device=device,
        )
    )
    return A, B, W


def make_layout(
    n_x: int,
    n_u: int,
    n_w: int,
    N: int,
) -> tuple[list[slice], list[slice], list[tuple[int, int, int, int]], int]:
    """Return variable slices and the free strictly causal K-entry index list."""
    offset = 0
    mu_slices = []
    for _ in range(N + 1):
        mu_slices.append(slice(offset, offset + n_x))
        offset += n_x

    v_slices = []
    for _ in range(N):
        v_slices.append(slice(offset, offset + n_u))
        offset += n_u

    k_entries = []
    for i in range(1, N):
        for j in range(i):
            for u_idx in range(n_u):
                for w_idx in range(n_w):
                    k_entries.append((i, j, u_idx, w_idx))
                    offset += 1

    return mu_slices, v_slices, k_entries, offset


def k_basis_tensor(
    entries: list[tuple[int, int, int, int]],
    N: int,
    n_u: int,
    n_w: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Float[Tensor, "n_k N_times_n_u N_times_n_w"]:
    """Stack dense basis matrices for all free strictly causal K scalars."""
    basis = torch.zeros(len(entries), N * n_u, N * n_w, dtype=dtype, device=device)
    if entries:
        rows = torch.tensor([i * n_u + u_idx for i, _j, u_idx, _w_idx in entries], device=device)
        cols = torch.tensor([j * n_w + w_idx for _i, j, _u_idx, w_idx in entries], device=device)
        basis[torch.arange(len(entries), device=device), rows, cols] = 1.0
    return basis


class CovarianceSteeringDynamics(Protocol):
    """Discrete-time dynamics interface used by covariance-steering SCP."""

    n_x: int
    n_u: int
    n_w: int

    def step(self, x: Float[Tensor, "n_x"], u: Float[Tensor, "n_u"], dt: float) -> Float[Tensor, "n_x"]:
        """Return the next state."""

    def linearize(
        self,
        x_bar: Float[Tensor, "n_x"],
        u_bar: Float[Tensor, "n_u"],
        dt: float,
    ) -> tuple[Float[Tensor, "n_x n_x"], Float[Tensor, "n_x n_u"], Float[Tensor, "n_x"]]:
        """Return `A, B, r` with `x+ ~= A x + B u + r`."""

    def disturbance(
        self,
        k: int,
        x_bar: Float[Tensor, "n_x"],
        u_bar: Float[Tensor, "n_u"],
        dt: float,
    ) -> tuple[Float[Tensor, "n_x n_w"], Float[Tensor, "n_w n_w"]]:
        """Return disturbance gain and covariance for step `k`."""


class DoubleIntegratorDynamics:
    """2D double integrator, exactly discretized (zero-order hold) with white acceleration noise."""

    n_x = 4
    n_u = 2
    n_w = 4

    def __init__(self, sigma_a: float = 0.1) -> None:
        self.sigma_a = float(sigma_a)

    def matrices(self, dt: float, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        return double_integrator_dynamics(dt, self.sigma_a, dtype, device)

    def step(self, x: Tensor, u: Tensor, dt: float) -> Tensor:
        A, B, _W = self.matrices(dt, x.dtype, x.device)
        return A @ x + B @ u

    def linearize(self, x_bar: Tensor, u_bar: Tensor, dt: float) -> tuple[Tensor, Tensor, Tensor]:
        A, B, _W = self.matrices(dt, x_bar.dtype, x_bar.device)
        r = self.step(x_bar, u_bar, dt) - A @ x_bar - B @ u_bar
        return A, B, r

    def disturbance(self, k: int, x_bar: Tensor, u_bar: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        del k, u_bar
        _A, _B, W = self.matrices(dt, x_bar.dtype, x_bar.device)
        return torch.eye(4, dtype=x_bar.dtype, device=x_bar.device), W


class UnicycleDynamics:
    """Euler-discrete kinematic unicycle dynamics."""

    n_x = 3
    n_u = 2
    n_w = 3

    def __init__(self, process_std: tuple[float, float, float] = (0.02, 0.02, 0.01)) -> None:
        self.process_std = tuple(float(v) for v in process_std)

    def step(self, x: Tensor, u: Tensor, dt: float) -> Tensor:
        theta = x[2]
        v = u[0]
        omega = u[1]
        return x + dt * torch.stack([v * torch.cos(theta), v * torch.sin(theta), omega])

    def linearize(self, x_bar: Tensor, u_bar: Tensor, dt: float) -> tuple[Tensor, Tensor, Tensor]:
        theta = x_bar[2]
        v = u_bar[0]
        c = torch.cos(theta)
        s = torch.sin(theta)
        A = torch.eye(3, dtype=x_bar.dtype, device=x_bar.device)
        A[0, 2] = -dt * v * s
        A[1, 2] = dt * v * c
        B = torch.zeros(3, 2, dtype=x_bar.dtype, device=x_bar.device)
        B[0, 0] = dt * c
        B[1, 0] = dt * s
        B[2, 1] = dt
        r = self.step(x_bar, u_bar, dt) - A @ x_bar - B @ u_bar
        return A, B, r

    def disturbance(self, k: int, x_bar: Tensor, u_bar: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        del k, u_bar
        std = torch.as_tensor(self.process_std, dtype=x_bar.dtype, device=x_bar.device)
        return torch.eye(3, dtype=x_bar.dtype, device=x_bar.device), torch.diag((std**2) * dt)


@dataclass
class CovarianceSteeringInstance:
    """Problem data for one nonlinear covariance-steering SCP instance."""

    mu_0: Float[Tensor, "n_x"]
    mu_goal: Float[Tensor, "n_x"]
    N: int
    dt: float
    Q: Float[Tensor, "n_x n_x"]
    R: Float[Tensor, "n_u n_u"]
    Q_N: Float[Tensor, "n_x n_x"]
    Sigma_0: Float[Tensor, "n_x n_x"]
    Sigma_N: Float[Tensor, "n_x n_x"] | None = None
    Sigma_max: Float[Tensor, "n_x n_x"] | None = None
    lambda_nu: float = 1e3


@dataclass
class SCP_Config:
    """SCP hyperparameters."""

    max_iters: int = 15
    trust_x0: float = 5.0
    trust_u0: float = 5.0
    trust_x_min: float = 1e-3
    trust_u_min: float = 1e-3
    trust_x_max: float = 50.0
    trust_u_max: float = 50.0
    rho0: float = 0.0
    rho1: float = 0.7
    shrink: float = 0.5
    expand: float = 2.0
    kkt_tol: float = 1e-3
    solver: str = "clarabel"
    solver_kwargs: dict | None = None
    policy: HyperparameterPolicy | None = None
    # Apply Ruiz equilibration to each subproblem before solving.
    # The SCP trust-region loop unscales the solution internally.
    equilibrate: bool = False


@dataclass
class SCP_Residuals:
    """Residuals used by the nonlinear SCP loop."""

    solver_residual: float
    dynamics_defect_inf: float
    virtual_control_l1: float
    state_change_inf: float
    input_change_inf: float
    overall_inf: float


@dataclass
class SCP_Iterate:
    """One solved covariance-steering SCP subproblem."""

    iter_idx: int
    P: Tensor
    q: Tensor
    A: Tensor
    b: Tensor
    cones: list[Cone]
    result: Solver_Result
    accepted: bool
    converged: bool
    residuals: SCP_Residuals
    formulate_time: float = 0.0
    solve_time: float = 0.0


class SCP_Failure(RuntimeError):
    """Raised when an SCP subproblem cannot be solved."""


def build_ltv_lifted_operators(
    A_seq: Float[Tensor, "N n_x n_x"],
    B_seq: Float[Tensor, "N n_x n_u"],
    D_seq: Float[Tensor, "N n_x n_w"],
) -> tuple[Tensor, Tensor, Tensor]:
    """Build lifted operators for a discrete-time LTV system."""
    N = A_seq.shape[0]
    n_x = A_seq.shape[1]
    n_u = B_seq.shape[2]
    n_w = D_seq.shape[2]
    tk = {"dtype": A_seq.dtype, "device": A_seq.device}

    Phi = [[torch.eye(n_x, **tk) for _ in range(N + 1)] for _ in range(N + 1)]
    for k in range(1, N + 1):
        for j in range(k):
            prod = torch.eye(n_x, **tk)
            for ell in range(j, k):
                prod = A_seq[ell] @ prod
            Phi[k][j] = prod

    A_bar = torch.zeros((N + 1) * n_x, n_x, **tk)
    B_bar = torch.zeros((N + 1) * n_x, N * n_u, **tk)
    D_bar = torch.zeros((N + 1) * n_x, N * n_w, **tk)
    for k in range(N + 1):
        A_bar[k * n_x : (k + 1) * n_x, :] = Phi[k][0]
        for j in range(k):
            transition = Phi[k][j + 1]
            B_bar[k * n_x : (k + 1) * n_x, j * n_u : (j + 1) * n_u] = transition @ B_seq[j]
            D_bar[k * n_x : (k + 1) * n_x, j * n_w : (j + 1) * n_w] = transition @ D_seq[j]
    return A_bar, B_bar, D_bar


def covariance_lmi_data_ltv(
    step: int,
    M: Tensor,
    A_bar: Tensor,
    B_bar: Tensor,
    D_bar: Tensor,
    Sigma_0: Tensor,
    Sigma_W_sqrt: Tensor,
    k_basis: Tensor,
    n_x: int,
    n_w_total: int,
) -> tuple[Tensor, Tensor]:
    """Return `svec(S_const)` and the gain columns of the covariance LMI at one step.

    The bound `Cov(x_step) <= M` is encoded by its Schur complement, which is affine in `K`
    (`A_k`, `B_k`, `D_k` are the step-`k` rows of the lifted dynamics):
        [     M - A_k Sigma_0 A_k',       (B_k K + D_k) Sigma_W^{1/2}  ]
        [ ((B_k K + D_k) Sigma_W^{1/2})',              I               ]  >= 0.
    """
    row = slice(step * n_x, (step + 1) * n_x)
    d_lmi_mat = n_x + n_w_total
    tk = {"dtype": A_bar.dtype, "device": A_bar.device}

    S_const = torch.zeros(d_lmi_mat, d_lmi_mat, **tk)
    A_step = A_bar[row, :]
    D_step = D_bar[row, :]
    S_const[:n_x, :n_x] = M - A_step @ Sigma_0 @ A_step.mT
    S_const[:n_x, n_x:] = D_step @ Sigma_W_sqrt
    S_const[n_x:, :n_x] = S_const[:n_x, n_x:].mT
    S_const[n_x:, n_x:] = torch.eye(n_w_total, **tk)

    if k_basis.shape[0] == 0:
        return svec(S_const), torch.zeros(svec(S_const).numel(), 0, **tk)

    offdiag = torch.einsum("ij,ajk,kl->ail", B_bar[row, :], k_basis, Sigma_W_sqrt)
    dS = torch.zeros(k_basis.shape[0], d_lmi_mat, d_lmi_mat, **tk)
    dS[:, :n_x, n_x:] = offdiag
    dS[:, n_x:, :n_x] = offdiag.transpose(1, 2)
    return svec(S_const), svec(dS).mT


def linearize_reference(
    dynamics: CovarianceSteeringDynamics,
    instance: CovarianceSteeringInstance,
    mu_ref: Tensor,
    u_ref: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    A_parts = []
    B_parts = []
    r_parts = []
    D_parts = []
    W_parts = []
    for k in range(instance.N):
        A_k, B_k, r_k = dynamics.linearize(mu_ref[k], u_ref[k], instance.dt)
        D_k, W_k = dynamics.disturbance(k, mu_ref[k], u_ref[k], instance.dt)
        A_parts.append(A_k)
        B_parts.append(B_k)
        r_parts.append(r_k)
        D_parts.append(D_k)
        W_parts.append(W_k)
    return torch.stack(A_parts), torch.stack(B_parts), torch.stack(r_parts), torch.stack(D_parts), torch.stack(W_parts)


def build_scp_subproblem(
    dynamics: CovarianceSteeringDynamics,
    instance: CovarianceSteeringInstance,
    mu_ref: Float[Tensor, "N_plus_1 n_x"],
    u_ref: Float[Tensor, "N n_u"],
    *,
    trust_x: float,
    trust_u: float,
    layout: str = "csr",
    psd_gradient_method: str = "dk",
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[Cone], dict[str, slice]]:
    """Build one nonlinear covariance-steering SCP subproblem about the reference `(mu_ref, u_ref)`."""
    A_seq, B_seq, r_seq, D_seq, W_seq = linearize_reference(dynamics, instance, mu_ref, u_ref)
    return build_scp_subproblem_from_linearization(
        instance,
        A_seq,
        B_seq,
        r_seq,
        D_seq,
        W_seq,
        mu_ref,
        u_ref,
        trust_x=trust_x,
        trust_u=trust_u,
        layout=layout,
        psd_gradient_method=psd_gradient_method,
    )


def build_scp_subproblem_from_linearization(
    instance: CovarianceSteeringInstance,
    A_seq: Float[Tensor, "N n_x n_x"],
    B_seq: Float[Tensor, "N n_x n_u"],
    r_seq: Float[Tensor, "N n_x"],
    D_seq: Float[Tensor, "N n_x n_w"],
    W_seq: Float[Tensor, "N n_w n_w"],
    mu_ref: Float[Tensor, "N_plus_1 n_x"],
    u_ref: Float[Tensor, "N n_u"],
    *,
    trust_x: float,
    trust_u: float,
    layout: str = "csr",
    psd_gradient_method: str = "dk",
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[Cone], dict[str, slice]]:
    """Build one SCP subproblem from precomputed LTV linearization data.

    The subproblem minimizes the expected quadratic cost of the mean trajectory
    and the disturbance-feedback response, along with an L1 penalty `lambda_nu ||nu||_1`
    on the virtual control that relaxes the linearized dynamics.

    The constraints include:
      - the initial mean, terminal mean, and linearized mean dynamics (zero cone)
      - the nonnegativity of the virtual control (nonnegative orthant)
      - the covariance bounds (PSD cones)
      - the trust regions (second-order cones)

    Returns `(P, q, A, b, cones, inds)`, where `inds` maps variable names to their slices.
    """
    N = instance.N
    n_x = A_seq.shape[1]
    n_u = B_seq.shape[2]
    n_w = D_seq.shape[2]
    dtype = A_seq.dtype
    device = A_seq.device
    tk = {"dtype": dtype, "device": device}

    mu_slices, v_slices, k_entries, base_dec = make_layout(n_x, n_u, n_w, N)
    nu_pos_slices = []
    nu_neg_slices = []
    offset = base_dec
    for _ in range(N):
        nu_pos_slices.append(slice(offset, offset + n_x))
        offset += n_x
    for _ in range(N):
        nu_neg_slices.append(slice(offset, offset + n_x))
        offset += n_x
    n_dec = offset
    k_start = v_slices[-1].stop if v_slices else mu_slices[-1].stop
    k_slice = slice(k_start, base_dec)

    inds: dict[str, slice] = {"K": k_slice}
    for k, sl in enumerate(mu_slices):
        inds[f"mu_{k}"] = sl
    for k, sl in enumerate(v_slices):
        inds[f"u_{k}"] = sl
    for k, sl in enumerate(nu_pos_slices):
        inds[f"nu_pos_{k}"] = sl
    for k, sl in enumerate(nu_neg_slices):
        inds[f"nu_neg_{k}"] = sl

    A_bar, B_bar, D_bar = build_ltv_lifted_operators(A_seq, B_seq, D_seq)
    Sigma_W = torch.block_diag(*[W_seq[k] for k in range(N)])
    Sigma_W_sqrt = torch.linalg.cholesky(Sigma_W)
    n_w_total = N * n_w
    k_basis = k_basis_tensor(k_entries, N, n_u, n_w, dtype, device)

    P = torch.zeros(n_dec, n_dec, **tk)
    q = torch.zeros(n_dec, **tk)
    for k in range(N):
        P[mu_slices[k], mu_slices[k]] = 2.0 * instance.Q
        P[v_slices[k], v_slices[k]] = 2.0 * instance.R
        q[nu_pos_slices[k]] = instance.lambda_nu
        q[nu_neg_slices[k]] = instance.lambda_nu
    P[mu_slices[N], mu_slices[N]] = 2.0 * instance.Q_N

    if k_entries:
        n_k = len(k_entries)
        k_quad = torch.zeros(n_k, n_k, **tk)
        k_lin = torch.zeros(n_k, **tk)
        for step in range(N + 1):
            Q_step = instance.Q_N if step == N else instance.Q
            row = slice(step * n_x, (step + 1) * n_x)
            F_const = D_bar[row, :]
            F_basis = torch.einsum("ij,ajk->aik", B_bar[row, :], k_basis)
            k_lin += 2.0 * torch.einsum("ij,ajk,kl,il->a", Q_step, F_basis, Sigma_W, F_const)
            k_quad += torch.einsum("ij,ajk,kl,bil->ab", Q_step, F_basis, Sigma_W, F_basis)
        for step in range(N):
            row = slice(step * n_u, (step + 1) * n_u)
            U_basis = k_basis[:, row, :]
            k_quad += torch.einsum("ij,ajk,kl,bil->ab", instance.R, U_basis, Sigma_W, U_basis)
        P[k_slice, k_slice] = 2.0 * (0.5 * (k_quad + k_quad.mT))
        q[k_slice] = k_lin

    A_rows: list[Tensor] = []
    b_parts: list[Tensor] = []
    cone_specs: list[tuple[type[Cone], int, dict]] = []

    def add_block(A_block: Tensor, b_block: Tensor, cone_type: type[Cone], **cone_kwargs) -> None:
        A_rows.append(A_block)
        b_parts.append(b_block)
        cone_specs.append((cone_type, b_block.numel(), cone_kwargs))

    A_block = torch.zeros(n_x, n_dec, **tk)
    A_block[:, mu_slices[0]] = torch.eye(n_x, **tk)
    add_block(A_block, instance.mu_0.clone(), ZeroCone)

    for k in range(N):
        A_block = torch.zeros(n_x, n_dec, **tk)
        A_block[:, mu_slices[k + 1]] = torch.eye(n_x, **tk)
        A_block[:, mu_slices[k]] = -A_seq[k]
        A_block[:, v_slices[k]] = -B_seq[k]
        A_block[:, nu_pos_slices[k]] = -torch.eye(n_x, **tk)
        A_block[:, nu_neg_slices[k]] = torch.eye(n_x, **tk)
        add_block(A_block, r_seq[k].clone(), ZeroCone)

    A_block = torch.zeros(n_x, n_dec, **tk)
    A_block[:, mu_slices[N]] = torch.eye(n_x, **tk)
    add_block(A_block, instance.mu_goal.clone(), ZeroCone)

    if N > 0:
        A_block = torch.zeros(2 * N * n_x, n_dec, **tk)
        for k in range(N):
            r0 = k * n_x
            A_block[r0 : r0 + n_x, nu_pos_slices[k]] = -torch.eye(n_x, **tk)
            r1 = (N + k) * n_x
            A_block[r1 : r1 + n_x, nu_neg_slices[k]] = -torch.eye(n_x, **tk)
        add_block(A_block, torch.zeros(2 * N * n_x, **tk), NonnegativeOrthant)

    lmi_targets: list[tuple[int, Tensor]] = []
    if instance.Sigma_max is not None:
        lmi_targets.extend((k, instance.Sigma_max) for k in range(1, N + 1))
    if instance.Sigma_N is not None:
        lmi_targets.append((N, instance.Sigma_N))

    for step, target in lmi_targets:
        b_lmi, cols_lmi = covariance_lmi_data_ltv(
            step, target, A_bar, B_bar, D_bar, instance.Sigma_0, Sigma_W_sqrt, k_basis, n_x, n_w_total
        )
        A_block = torch.zeros(b_lmi.numel(), n_dec, **tk)
        A_block[:, k_slice] = -cols_lmi
        add_block(A_block, b_lmi, PositiveSemidefiniteCone, gradient_method=psd_gradient_method)

    for k in range(N + 1):
        A_block = torch.zeros(1 + n_x, n_dec, **tk)
        b_block = torch.zeros(1 + n_x, **tk)
        b_block[0] = trust_x
        A_block[1:, mu_slices[k]] = torch.eye(n_x, **tk)
        b_block[1:] = mu_ref[k]
        add_block(A_block, b_block, SecondOrderCone)
    for k in range(N):
        A_block = torch.zeros(1 + n_u, n_dec, **tk)
        b_block = torch.zeros(1 + n_u, **tk)
        b_block[0] = trust_u
        A_block[1:, v_slices[k]] = torch.eye(n_u, **tk)
        b_block[1:] = u_ref[k]
        add_block(A_block, b_block, SecondOrderCone)

    A = torch.cat(A_rows, dim=0)
    b = torch.cat(b_parts, dim=0)

    cones: list[Cone] = []
    row_cursor = 0
    for cone_type, size, kwargs in cone_specs:
        idx = torch.arange(row_cursor, row_cursor + size, device=device)
        cones.append(cone_type(idx, **kwargs))
        row_cursor += size

    if layout == "coo":
        P = P.to_sparse_coo()
        A = A.to_sparse_coo()
    if layout == "csr":
        P = P.to_sparse_csr()
        A = A.to_sparse_csr()
    if layout == "csc":
        P = P.to_sparse_csc()
        A = A.to_sparse_csc()
    else:  # layout == "dense"
        pass

    return P, q, A, b, cones, inds


def initialize_reference(
    dynamics: CovarianceSteeringDynamics,
    instance: CovarianceSteeringInstance,
) -> tuple[Tensor, Tensor]:
    alphas = torch.linspace(0.0, 1.0, instance.N + 1, dtype=instance.mu_0.dtype, device=instance.mu_0.device).unsqueeze(
        -1
    )
    mu_ref = (1.0 - alphas) * instance.mu_0 + alphas * instance.mu_goal
    u_ref = torch.zeros(instance.N, dynamics.n_u, dtype=instance.mu_0.dtype, device=instance.mu_0.device)
    if isinstance(dynamics, UnicycleDynamics):
        delta = instance.mu_goal[:2] - instance.mu_0[:2]
        distance = torch.linalg.vector_norm(delta).item()
        heading = torch.atan2(delta[1], delta[0]) if distance > 1e-12 else instance.mu_0[2]
        mu_ref[:, 2] = heading
        u_ref[:, 0] = distance / max(instance.N * instance.dt, 1e-12)
    return mu_ref, u_ref


def decode_solution(x_opt: Tensor, inds: dict[str, slice], N: int) -> tuple[Tensor, Tensor, Tensor]:
    mu = torch.stack([x_opt[inds[f"mu_{k}"]] for k in range(N + 1)])
    u = torch.stack([x_opt[inds[f"u_{k}"]] for k in range(N)])
    nu = torch.stack([x_opt[inds[f"nu_pos_{k}"]] - x_opt[inds[f"nu_neg_{k}"]] for k in range(N)])
    return mu, u, nu


def trajectory_cost(instance: CovarianceSteeringInstance, mu: Tensor, u: Tensor, nu: Tensor) -> float:
    cost = torch.zeros((), dtype=mu.dtype, device=mu.device)
    for k in range(instance.N):
        cost = cost + mu[k] @ instance.Q @ mu[k] + u[k] @ instance.R @ u[k]
    cost = cost + mu[instance.N] @ instance.Q_N @ mu[instance.N]
    cost = cost + instance.lambda_nu * torch.linalg.vector_norm(nu.reshape(-1), ord=1)
    return float(cost.item())


def compute_residuals(
    dynamics: CovarianceSteeringDynamics,
    instance: CovarianceSteeringInstance,
    result: Solver_Result,
    inds: dict[str, slice],
    mu_prev: Tensor,
    u_prev: Tensor,
    x_opt: Tensor | None = None,
) -> SCP_Residuals:
    # `x_opt` lets callers pass the solution in the original (unequilibrated) space.
    # The solver residual norms below are taken from the result as-is.
    mu, u, nu = decode_solution(result.x if x_opt is None else x_opt, inds, instance.N)
    defect = 0.0
    for k in range(instance.N):
        step_defect = mu[k + 1] - dynamics.step(mu[k], u[k], instance.dt) - nu[k]
        defect = max(defect, float(torch.max(torch.abs(step_defect)).item()))
    virtual_l1 = float(torch.linalg.vector_norm(nu.reshape(-1), ord=1).item())
    state_change = float(torch.max(torch.abs(mu - mu_prev)).item())
    input_change = float(torch.max(torch.abs(u - u_prev)).item()) if instance.N else 0.0
    solver_residual = 0.0
    if result.dual_res_norm_hist.numel() and result.prim_res_norm_hist.numel():
        solver_residual = float(torch.maximum(result.dual_res_norm_hist[-1], result.prim_res_norm_hist[-1]).item())
    overall = max(defect, virtual_l1)
    return SCP_Residuals(
        solver_residual=solver_residual,
        dynamics_defect_inf=defect,
        virtual_control_l1=virtual_l1,
        state_change_inf=state_change,
        input_change_inf=input_change,
        overall_inf=overall,
    )


def make_persistent_solver(cfg: SCP_Config) -> ClarabelSCPSolver | None:
    """Build the stateful Clarabel solver for `cfg.solver == "clarabel_scp"` (else `None`).

    `ClarabelSCPSolver` caches Clarabel's equilibration and symbolic
    KKT factorization and reuses them across SCP outer iterations.
    """
    if cfg.solver != "clarabel_scp":
        return None
    return ClarabelSCPSolver(**(cfg.solver_kwargs or {}))


def solve_subproblem(
    P: Tensor,
    q: Tensor,
    A: Tensor,
    b: Tensor,
    cones: list[Cone],
    cfg: SCP_Config,
    clarabel_scp_solver: ClarabelSCPSolver | None = None,
) -> tuple[Solver_Result, tuple[Tensor, Tensor, Tensor] | None]:
    """Solve one SCP subproblem, optionally equilibrating it first.

    When `cfg.equilibrate` is set, `(P, q, A, b)` are equilibrated in-place.

    `cfg.solver` selects the backend:
      - "clarabel" builds a fresh native solver each call
      - "clarabel_scp" reuses the persistent "clarabel_scp_solver"
      - "cosmo" runs our differentiable solver
    """
    scaling = None
    if cfg.equilibrate:
        scaling = equilibriate_(P, q, A, b, cones, max_iter=10, conv_tol=0.0)

    kwargs = dict(cfg.solver_kwargs or {})
    if cfg.solver == "clarabel":
        x_opt, y_opt, obj_val, status = solve_with_clarabel(P, q, A, b, cones, **kwargs)
        if x_opt is None:
            raise SCP_Failure(f"Clarabel returned status '{status}'")
        return make_solver_result(x_opt, y_opt, A, b, obj_val), scaling
    if cfg.solver == "clarabel_scp":
        if clarabel_scp_solver is None:
            raise ValueError(
                "solver='clarabel_scp' needs a persistent ClarabelSCPSolver; "
                "use `solve_scp` rather than calling `solve_subproblem` directly."
            )
        x_opt, y_opt, obj_val, status, _stats = clarabel_scp_solver.solve_with_stats(P, q, A, b, cones)
        if x_opt is None:
            raise SCP_Failure(f"Clarabel (native) returned status '{status}'")
        return make_solver_result(x_opt, y_opt, A, b, obj_val), scaling
    if cfg.solver == "cosmo":
        solver = Solver(P, q, A, b, prox=ProjectionProx(cones), policy=cfg.policy or AdaptiveRhoOSQP())
        result = solver.solve(**kwargs)
        if not result.converged:
            raise SCP_Failure(f"Solver did not converge; status={result.status}")
        return result, scaling
    raise ValueError("scp solver must be 'clarabel', 'clarabel_scp', or 'cosmo'.")


def unscale_solution(result: Solver_Result, scaling: tuple[Tensor, Tensor, Tensor] | None) -> tuple[Tensor, float]:
    """Unscale an equilibrated solution back to the original problem coordinates."""
    obj_val = float(result.obj_val_hist[-1].item())
    if scaling is None:
        return result.x, obj_val
    d, _e, c = scaling
    return d * result.x, obj_val / float(c)


def solve_scp(
    dynamics: CovarianceSteeringDynamics,
    instance: CovarianceSteeringInstance,
    scp_cfg: SCP_Config | None = None,
) -> Iterator[SCP_Iterate]:
    """Run unconstrained covariance-steering SCP and yield solved subproblems."""
    cfg = scp_cfg if scp_cfg is not None else SCP_Config()
    mu_ref, u_ref = initialize_reference(dynamics, instance)
    trust_x = cfg.trust_x0
    trust_u = cfg.trust_u0
    J_ref = trajectory_cost(instance, mu_ref, u_ref, torch.zeros(instance.N, dynamics.n_x, dtype=mu_ref.dtype))
    clarabel_scp_solver = make_persistent_solver(cfg)

    for i in range(cfg.max_iters):
        formulate_start = time.perf_counter()
        P, q, A, b, cones, inds = build_scp_subproblem(
            dynamics, instance, mu_ref, u_ref, trust_x=trust_x, trust_u=trust_u
        )
        formulate_time = time.perf_counter() - formulate_start

        solve_start = time.perf_counter()
        result, scaling = solve_subproblem(P, q, A, b, cones, cfg, clarabel_scp_solver)
        solve_time = time.perf_counter() - solve_start

        x_opt, obj_val = unscale_solution(result, scaling)
        mu_new, u_new, nu_new = decode_solution(x_opt, inds, instance.N)
        J_new = trajectory_cost(instance, mu_new, u_new, nu_new)
        predicted = J_ref - obj_val
        actual = J_ref - J_new
        rho = 1.0 if abs(predicted) < 1e-12 else actual / predicted
        accepted = rho >= cfg.rho0
        residuals = compute_residuals(dynamics, instance, result, inds, mu_ref, u_ref, x_opt=x_opt)
        converged = accepted and residuals.overall_inf < cfg.kkt_tol

        yield SCP_Iterate(i, P, q, A, b, cones, result, accepted, converged, residuals, formulate_time, solve_time)

        if converged:
            return
        if accepted:
            mu_ref, u_ref = mu_new, u_new
            J_ref = J_new
            if rho > cfg.rho1:
                trust_x = min(cfg.expand * trust_x, cfg.trust_x_max)
                trust_u = min(cfg.expand * trust_u, cfg.trust_u_max)
        else:
            trust_x = max(cfg.shrink * trust_x, cfg.trust_x_min)
            trust_u = max(cfg.shrink * trust_u, cfg.trust_u_min)


def uniform(low: float, high: float, shape: tuple[int, ...] = (), *, dtype=None, device=None) -> Tensor:
    """Draw uniform samples in `[low, high)` from the global torch RNG (seed-controlled)."""
    return torch.empty(shape, dtype=dtype, device=device).uniform_(low, high)


def resolve_std(value: float | None, value_range: tuple[float, float], *, dtype, device) -> float:
    """Return `value` if pinned, else a uniform sample from `value_range`."""
    if value is not None:
        return float(value)
    return uniform(value_range[0], value_range[1], dtype=dtype, device=device).item()


def sample_double_integrator_instance(
    N: int = 10,
    dt: float = 0.5,
    sigma_a: float = 0.1,
    start: tuple[float, float] | None = None,
    goal: tuple[float, float] | None = None,
    goal_box: tuple[float, float] = (10.0, 10.0),
    sigma_0: float | None = None,
    sigma_N: float | None = None,
    sigma_0_range: tuple[float, float] = (0.005, 0.02),
    sigma_N_range: tuple[float, float] = (0.1, 0.4),
    q_p: float = 1.0,
    q_v: float = 0.3,
    r: float = 0.1,
) -> tuple[DoubleIntegratorDynamics, CovarianceSteeringInstance]:
    """Sample a random double-integrator covariance-steering instance.

    The goal is sampled from a box centered on the start.
    The boundary stds are sampled from `sigma_0_range` / `sigma_N_range`.
    Passing `start` / `goal` / `sigma_0` / `sigma_N` explicitly disables randomization for that quantity.
    """

    dtype = torch.get_default_dtype()
    device = torch.get_default_device()
    dyn = DoubleIntegratorDynamics(sigma_a=sigma_a)

    if start is None:
        start_xy = torch.zeros(2, dtype=dtype, device=device)
    else:
        start_xy = torch.tensor(start, dtype=dtype, device=device)
    if goal is None:
        half = 0.5 * torch.tensor(goal_box, dtype=dtype, device=device)
        goal_xy = start_xy + (2.0 * torch.rand(2, dtype=dtype, device=device) - 1.0) * half
    else:
        goal_xy = torch.tensor(goal, dtype=dtype, device=device)

    mu_0 = torch.cat([start_xy, torch.zeros(2, dtype=dtype, device=device)])
    mu_goal = torch.cat([goal_xy, torch.zeros(2, dtype=dtype, device=device)])

    s0 = resolve_std(sigma_0, sigma_0_range, dtype=dtype, device=device)
    sN = resolve_std(sigma_N, sigma_N_range, dtype=dtype, device=device)

    Q = torch.diag(torch.tensor([q_p, q_p, q_v, q_v], dtype=dtype, device=device))
    R = r * torch.eye(2, dtype=dtype, device=device)
    inst = CovarianceSteeringInstance(
        mu_0=mu_0,
        mu_goal=mu_goal,
        N=N,
        dt=dt,
        Q=Q,
        R=R,
        Q_N=10.0 * Q,
        Sigma_0=(s0**2) * torch.eye(4, dtype=dtype, device=device),
        Sigma_N=(sN**2) * torch.eye(4, dtype=dtype, device=device),
        lambda_nu=1e4,
    )
    return dyn, inst


def sample_unicycle_instance(
    N: int = 30,
    dt: float = 0.1,
    start: tuple[float, float, float] | None = None,
    goal: tuple[float, float, float] | None = None,
    goal_box: tuple[float, float] = (10.0, 10.0),
    process_std: tuple[float, float, float] = (0.02, 0.02, 0.01),
    sigma_0: float | None = None,
    sigma_N: float | None = None,
    sigma_0_range: tuple[float, float] = (0.01, 0.05),
    sigma_N_range: tuple[float, float] = (0.05, 0.25),
    sigma_max: float | None = None,
    lambda_nu: float = 1e4,
) -> tuple[UnicycleDynamics, CovarianceSteeringInstance]:
    """Sample a random unicycle covariance-steering instance.

    The goal is sampled from a box centered on the start.
    Headings are sampled uniformly from `[-pi, pi]` for both the start and goal.
    The boundary stds are sampled from `sigma_0_range` / `sigma_N_range`.
    Passing `start` / `goal` / `sigma_0` / `sigma_N` explicitly disables randomization for that quantity.
    """

    dtype = torch.get_default_dtype()
    device = torch.get_default_device()
    dyn = UnicycleDynamics(process_std=process_std)

    if start is None:
        theta_0 = uniform(-torch.pi, torch.pi, dtype=dtype, device=device).reshape(1)
        mu_0 = torch.cat([torch.zeros(2, dtype=dtype, device=device), theta_0])
    else:
        mu_0 = torch.tensor(start, dtype=dtype, device=device)
    if goal is None:
        half = 0.5 * torch.tensor(goal_box, dtype=dtype, device=device)
        offset = (2.0 * torch.rand(2, dtype=dtype, device=device) - 1.0) * half
        theta_goal = uniform(-torch.pi, torch.pi, dtype=dtype, device=device).reshape(1)
        mu_goal = torch.cat([mu_0[:2] + offset, theta_goal])
    else:
        mu_goal = torch.tensor(goal, dtype=dtype, device=device)

    s0 = resolve_std(sigma_0, sigma_0_range, dtype=dtype, device=device)
    sN = resolve_std(sigma_N, sigma_N_range, dtype=dtype, device=device)

    Q = torch.diag(torch.tensor([1.0, 1.0, 0.1], dtype=dtype, device=device))
    R = torch.diag(torch.tensor([0.1, 0.05], dtype=dtype, device=device))
    Sigma_max = None
    if sigma_max is not None:
        Sigma_max = (sigma_max**2) * torch.eye(3, dtype=dtype, device=device)
    inst = CovarianceSteeringInstance(
        mu_0=mu_0,
        mu_goal=mu_goal,
        N=N,
        dt=dt,
        Q=Q,
        R=R,
        Q_N=10.0 * Q,
        Sigma_0=(s0**2) * torch.eye(3, dtype=dtype, device=device),
        Sigma_N=(sN**2) * torch.eye(3, dtype=dtype, device=device),
        Sigma_max=Sigma_max,
        lambda_nu=lambda_nu,
    )
    return dyn, inst
