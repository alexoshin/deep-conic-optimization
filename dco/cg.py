"""Batched preconditioned conjugate gradient on a linear operator."""

from collections.abc import Callable

from jaxtyping import Float
import torch
from torch import Tensor

from .linalg import dot_prod, matvec


class LinearOperator:
    """A symmetric linear map given by its matrix-vector product."""

    def __init__(self, A: Float[Tensor, "... n n"]):
        self.A = A

    def __call__(self, x: Float[Tensor, "... n"]) -> Float[Tensor, "... n"]:
        return self.matvec(x)

    def matvec(self, x: Float[Tensor, "... n"]) -> Float[Tensor, "... n"]:
        return matvec(self.A, x)


Preconditioner = Callable[[Float[Tensor, "... n"]], Float[Tensor, "... n"]]


def identity_preconditioner(r: Float[Tensor, "... n"]) -> Float[Tensor, "... n"]:
    return r


def jacobi_preconditioner(diag: Float[Tensor, "... n"]) -> Preconditioner:
    """Jacobi preconditioner `r -> r / diag`."""
    inv_diag = 1.0 / diag

    def precondition(r: Float[Tensor, "... n"]) -> Float[Tensor, "... n"]:
        return inv_diag * r

    return precondition


def conjugate_gradient(
    A: LinearOperator,
    b: Float[Tensor, "... n"],
    x0: Float[Tensor, "... n"] | None = None,
    preconditioner: Preconditioner = identity_preconditioner,
    max_iter: int | None = None,
    eps_abs: float = 0.0,
    eps_rel: float = 3e-4,
    tol_ref_sq: Float[Tensor, "..."] | None = None,
    check_every: int = 1,
) -> tuple[Float[Tensor, "... n"], int]:
    """Solve `Ax = b` for a symmetric positive definite `A`.

    Supports arbitrary leading batch dimensions.
    The loop runs until all systems reach `||r||_{M^{-1}}^2 <= eps_abs^2 + eps_rel^2 * ref_sq`,
    where `ref_sq` is the initial residual norm squared or `tol_ref_sq` when given.
    A warm start `x0` should pass `tol_ref_sq = b^T M^{-1} b`
    so that it changes the starting point and not the stopping rule.

    `check_every` is the iterations between convergence checks.
    The check costs one host synchronization.
    Systems that have already converged are frozen between checks
    so their returned iterate is the one they would have stopped at.

    Returns the solution and the number of iterations run.
    """

    n = b.shape[-1]
    if x0 is None:
        x0 = torch.zeros_like(b)
    if max_iter is None:
        max_iter = 4 * n

    x = x0
    r = b - A(x)
    d = preconditioner(r)
    r_norm_sq = dot_prod(r, d)
    conv_tol_sq = eps_abs**2 + eps_rel**2 * (r_norm_sq if tol_ref_sq is None else tol_ref_sq)
    freeze_converged = check_every != 1
    zero = torch.zeros((), dtype=b.dtype, device=b.device)

    k = 0
    for k in range(max_iter):
        converged = r_norm_sq <= conv_tol_sq
        if check_every > 0 and (k + 1) % check_every == 0 and torch.all(converged):
            break
        q = A(d)
        # d^T A d = 0 means no further progress can be made on that system
        alpha = torch.nan_to_num(r_norm_sq / dot_prod(d, q), nan=0.0)
        if freeze_converged:
            alpha = torch.where(converged, zero, alpha)
        x = x + alpha.unsqueeze(-1) * d
        r = r - alpha.unsqueeze(-1) * q
        s = preconditioner(r)
        r_norm_sq_prev = r_norm_sq
        r_norm_sq = dot_prod(r, s)
        beta = torch.nan_to_num(r_norm_sq / r_norm_sq_prev, nan=0.0)
        if freeze_converged:
            beta = torch.where(converged, zero, beta)
        d = s + beta.unsqueeze(-1) * d

    return x, k
