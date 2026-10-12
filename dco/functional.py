"""Functional interface for COSMO ADMM iteration steps.

`A` and `A^T` are modeled as linear operators (callables), so that the same code
can be used for both a single problem with trailing dimensions
and a flat super-batched problem with segment reducers.
"""

from collections.abc import Callable
from dataclasses import dataclass

from jaxtyping import Bool, Float
import torch
from torch import Tensor

# Bounds for the ADMM penalty parameter
RHO_MIN = 1e-6
RHO_MAX = 1e6


@dataclass
class Solver_Residuals:
    """Residuals of the conic program at one iterate."""

    obj_val: Float[Tensor, "..."]  # (1/2) x^T P x + q^T x
    obj_grad: Float[Tensor, "... n"]  # P x + q
    prim_res: Float[Tensor, "... m"]  # A x + z - b
    dual_res: Float[Tensor, "... n"]  # P x + q - A^T y  (== P x + q + A^T y*, y* the KKT multiplier)
    admm_dual_x: Float[Tensor, "... n"]  # x - x_prev
    admm_dual_z: Float[Tensor, "... m"]  # z - z_prev
    admm_prim_x: Float[Tensor, "... n"]  # x - x_tilde
    admm_prim_z: Float[Tensor, "... m"]  # z - z_tilde
    prim_rel_norm: Float[Tensor, "..."]  # max(||A x||_inf, ||z||_inf, ||b||_inf)
    dual_rel_norm: Float[Tensor, "..."]  # max(||P x||_inf, ||q||_inf, ||A^T y||_inf)


def clamp_rhos(rhos: Float[Tensor, "... m"]) -> Float[Tensor, "... m"]:
    return torch.clamp(rhos, min=RHO_MIN, max=RHO_MAX)


def kkt_rhs(
    sigma: float,
    x: Float[Tensor, "... n"],
    q: Float[Tensor, "... n"],
    b: Float[Tensor, "... m"],
    z: Float[Tensor, "... m"],
    y: Float[Tensor, "... m"],
    rhos: Float[Tensor, "... m"],
) -> Float[Tensor, "... n+m"]:
    """Compute the right-hand side of the quasidefinite KKT system `[sigma x - q; b - z + y / rho]`."""
    return torch.cat([sigma * x - q, b - z + y / rhos], dim=-1)


def reduced_rhs(
    sigma: float,
    x: Float[Tensor, "... n"],
    q: Float[Tensor, "... n"],
    b: Float[Tensor, "... m"],
    z: Float[Tensor, "... m"],
    y: Float[Tensor, "... m"],
    rhos: Float[Tensor, "... m"],
    matvec_AT: Callable[[Tensor], Tensor],
) -> Float[Tensor, "... n"]:
    """Compute the right-hand side of the reduced normal equations `sigma x - q + A^T (rho (b - z) + y)`."""
    return sigma * x - q + matvec_AT(rhos * (b - z) + y)


def recover_z_tilde(
    z: Float[Tensor, "... m"], nu: Float[Tensor, "... m"], y: Float[Tensor, "... m"], rhos: Float[Tensor, "... m"]
) -> Float[Tensor, "... m"]:
    """Recover `z_tilde = z - (nu + y) / rho` from the KKT dual variable."""
    return z - (nu + y) / rhos


def over_relax(alpha: Float[Tensor, "..."], new: Float[Tensor, "... k"], old: Float[Tensor, "... k"]) -> Tensor:
    return alpha * new + (1 - alpha) * old


def second_block(
    project: Callable[[Tensor], Tensor],
    x_tilde: Float[Tensor, "... n"],
    z_tilde: Float[Tensor, "... m"],
    x: Float[Tensor, "... n"],
    z: Float[Tensor, "... m"],
    y: Float[Tensor, "... m"],
    rhos: Float[Tensor, "... m"],
    alpha_x: Float[Tensor, "..."],
    alpha_z: Float[Tensor, "..."],
) -> tuple[Float[Tensor, "... n"], Float[Tensor, "... m"], Float[Tensor, "... m"]]:
    """Over-relax `x` and `z` and project the over-relaxed `z` onto the cone.

    Returns `(x_next, z_next, z_relaxed)`.
    """
    z_relaxed = over_relax(alpha_z, z_tilde, z)
    z_next = project(z_relaxed + y / rhos)
    return over_relax(alpha_x, x_tilde, x), z_next, z_relaxed


def dual_update(
    y: Float[Tensor, "... m"],
    rhos: Float[Tensor, "... m"],
    z_relaxed: Float[Tensor, "... m"],
    z: Float[Tensor, "... m"],
) -> Float[Tensor, "... m"]:
    """Compute the dual variable update `y + rho (alpha z_tilde + (1 - alpha) z_prev - z)`."""
    return y + rhos * (z_relaxed - z)


def admm_residuals(
    q: Float[Tensor, "... n"],
    b: Float[Tensor, "... m"],
    x: Float[Tensor, "... n"],
    z: Float[Tensor, "... m"],
    x_tilde: Float[Tensor, "... n"],
    z_tilde: Float[Tensor, "... m"],
    x_prev: Float[Tensor, "... n"],
    z_prev: Float[Tensor, "... m"],
    P_x: Float[Tensor, "... n"],
    A_x: Float[Tensor, "... m"],
    A_top_y: Float[Tensor, "... n"],
    dot_n: Callable[[Tensor, Tensor], Tensor],
    max_inf_norm_n: Callable[..., Tensor],
    max_inf_norm_m: Callable[..., Tensor],
) -> Solver_Residuals:
    """Compute the ADMM residuals of one iterate from the three matvecs `Px`, `Ax` and `A^T y`."""
    return Solver_Residuals(
        obj_val=0.5 * dot_n(x, P_x) + dot_n(q, x),
        obj_grad=P_x + q,
        # The iteration's dual lives in the polar cone, the negative of the KKT multiplier.
        dual_res=P_x + q - A_top_y,
        prim_res=A_x + z - b,
        admm_dual_x=x - x_prev,
        admm_dual_z=z - z_prev,
        admm_prim_x=x - x_tilde,
        admm_prim_z=z - z_tilde,
        prim_rel_norm=max_inf_norm_m(A_x, z, b),
        dual_rel_norm=max_inf_norm_n(P_x, q, A_top_y),
    )


def converged_flags(
    dual_res_norm: Float[Tensor, "..."],
    prim_res_norm: Float[Tensor, "..."],
    dual_rel_norm: Float[Tensor, "..."],
    prim_rel_norm: Float[Tensor, "..."],
    eps_abs: float,
    eps_rel: float,
) -> Bool[Tensor, "..."]:
    """Evaluate the convergence flags for each problem.

    Convergence is declared when both residual norms fall under `eps_abs + eps_rel * <their relative scale>`.
    """
    return (dual_res_norm <= eps_abs + eps_rel * dual_rel_norm) & (prim_res_norm <= eps_abs + eps_rel * prim_rel_norm)


def rho_change_gate(rhos: Float[Tensor, "... m"], rhos_new: Float[Tensor, "... m"], tolerance: float) -> Tensor:
    """Check whether `rhos_new` differs from `rhos` by more than the factor `tolerance` (a device scalar)."""
    return torch.any(rhos_new > tolerance * rhos) | torch.any(rhos_new < rhos / tolerance)
