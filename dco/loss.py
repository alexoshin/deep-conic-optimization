"""Loss for unrolled training."""

import math

from jaxtyping import Float
import torch
from torch import Tensor

from .data import Sample
from .linalg import L2_norm
from .policy import HyperparameterPolicy
from .problem import uniform_batch_metadata
from .solver import DEFAULT_CG_BACKWARD_EPS_ABS, DEFAULT_CG_BACKWARD_EPS_REL, ProjectionProx, Solver, Solver_Result
from .super_batch import BatchMetadata
from .warmstart import Initialization

METRICS = ("residuals", "residuals_sq")
NORMALIZATIONS = (None, "initial_gap")
TRANSFORMS = (None, "log", "log1p")


def compute_loss(
    batch_data: Sample,
    policy: HyperparameterPolicy,
    n_iters: int,
    initialization: Initialization | None = None,
    batch_metadata: BatchMetadata | None = None,
    sigma: float = 1e-6,
    adaptive_rho_tolerance: float = 1.0,
    linsys_solve_method: str = "indirect",
    cg_preconditioner: str = "jacobi",
    cg_max_iter: int | None = None,
    cg_eps_abs: float = 0.0,
    cg_eps_rel: float = 1e-8,
    cg_check_every: int = 1,
    cg_backward_eps_abs: float = DEFAULT_CG_BACKWARD_EPS_ABS,
    cg_backward_eps_rel: float = DEFAULT_CG_BACKWARD_EPS_REL,
    cg_backward_max_iter: int | None = None,
    adjoint_warm_start: bool = True,
    metric: str = "residuals_sq",
    all_iterates: bool = True,
    normalization: str | None = "initial_gap",
    transform: str | None = "log",
    discount_factor: float = 1.0,
    return_per_iterate: bool = False,
) -> tuple[Float[Tensor, "..."], Solver_Result, Float[Tensor, "... K+1"]]:
    """Unroll COSMO for `n_iters` iterations on `batch_data` and score the iterates.

    Loss options:
      metric:
        - residuals: `||r_d|| + ||r_p||`
        - residuals_sq: `||r_d||^2 + ||r_p||^2`
      normalization:
        - None: no normalization
        - initial_gap: divide each residual by its detached value at the first iterate
      transform:
        - None: no transform
        - log: apply `log` to each residual
        - log1p: apply `log1p` to each residual
      all_iterates:
        - True: sum over the unroll (weighted by `discount_factor^(K-k)`)
        - False: score the last iterate only

    The final loss is the per-problem sum over iterates, averaged over the batch.

    Returns `(loss, result, opt_gap_hist)`, where `opt_gap_hist` is `||x^k - x^*||_2 / sqrt(n)` (reporting only).
    When `return_per_iterate` is `True`, `loss` is the `(B, K+1)` per-iterate loss instead of a scalar.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got {normalization!r}")
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
    P, q, A, b, cones, x_opt, s_opt, y_opt = batch_data
    if batch_metadata is None and A.dim() > 2:
        m, n = A.shape[-2:]
        batch_metadata = uniform_batch_metadata(math.prod(A.shape[:-2]), n, m)

    prox = ProjectionProx(cones, batch_metadata=batch_metadata)
    solver = Solver(
        P, q, A, b, prox, policy=policy, initialization=initialization, sigma=sigma, batch_metadata=batch_metadata
    )
    result = solver.solve(
        max_iter=n_iters,
        eps_abs=0.0,
        eps_rel=0.0,
        eps_prim_inf=0.0,
        eps_dual_inf=0.0,
        check_termination=0,
        adaptive_rho_tolerance=adaptive_rho_tolerance,
        linsys_solve_method=linsys_solve_method,
        cg_preconditioner=cg_preconditioner,
        cg_max_iter=cg_max_iter,
        cg_eps_abs=cg_eps_abs,
        cg_eps_rel=cg_eps_rel,
        cg_check_every=cg_check_every,
        cg_backward_eps_abs=cg_backward_eps_abs,
        cg_backward_eps_rel=cg_backward_eps_rel,
        cg_backward_max_iter=cg_backward_max_iter,
        adjoint_warm_start=adjoint_warm_start,
        save_history=True,
    )

    squared = metric == "residuals_sq"

    def reduce(v: Tensor) -> Tensor:
        return torch.sum(v * v, dim=-1) if squared else L2_norm(v)

    # Flat (K+1, total) histories -> zero-padded (B, K+1, max) grids (padding is a no-op for these reductions)
    pad_n, pad_m = solver.batch_metadata.padders(solver.q.device)
    prim_flat = solver.matvec_A(result.x_hist) + result.z_hist - solver.b
    dual_flat = solver.matvec_P(result.x_hist) + solver.q + solver.matvec_AT(result.y_hist)
    prim = reduce(pad_m.to_pad(prim_flat)).transpose(-2, -1)  # (B, K+1)
    dual = reduce(pad_n.to_pad(dual_flat)).transpose(-2, -1)

    if normalization == "initial_gap":
        prim = prim / (prim[..., :1] + 1e-12).detach()
        dual = dual / (dual[..., :1] + 1e-12).detach()
    losses = prim + dual
    if transform == "log":
        losses = torch.log(losses + 1e-12)
    elif transform == "log1p":
        losses = torch.log1p(losses)
    per_iterate = losses

    if not all_iterates:
        losses = losses[..., -1:]
    elif discount_factor != 1.0:
        k = torch.arange(losses.shape[-1], device=losses.device)
        losses = losses * discount_factor ** (losses.shape[-1] - 1 - k)
    loss = losses.sum(dim=-1).mean()

    x_hist = pad_n.to_pad(result.x_hist).movedim(-2, 0)  # (B, K+1, max_n)
    x_opt_pp = pad_n.to_pad(x_opt.reshape(-1))
    n_div = torch.sqrt(pad_n.sizes_t.to(x_hist.dtype)).unsqueeze(-1)
    opt_gap_hist = L2_norm(x_hist - x_opt_pp.unsqueeze(-2)) / n_div
    return (per_iterate if return_per_iterate else loss), result, opt_gap_hist
