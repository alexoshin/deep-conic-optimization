"""Problem data equilibration routines."""

from jaxtyping import Float
import torch
from torch import Tensor

from .cones import Cone, NonnegativeOrthant, PositiveSemidefiniteCone, SecondOrderCone, ZeroCone


_SPARSE_LAYOUTS = (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)


def _is_sparse(A: Tensor) -> bool:
    return A.layout in _SPARSE_LAYOUTS


def _amax_or_zero(x: Tensor, dim: int) -> Tensor:
    other = -1 if dim == -2 else -2
    if x.shape[dim] == 0:
        return torch.zeros(x.shape[other], dtype=x.dtype, device=x.device)
    return torch.amax(torch.abs(x), dim=dim)


def _sparse_row_col_absmax(A: Tensor) -> tuple[Tensor, Tensor]:
    A_coo = A.coalesce() if A.layout == torch.sparse_coo else A.to_sparse_coo().coalesce()
    m, n = A.shape
    row_max = torch.zeros(m, dtype=A.dtype, device=A.device)
    col_max = torch.zeros(n, dtype=A.dtype, device=A.device)

    if A_coo._nnz() == 0:
        return row_max, col_max

    rows, cols = A_coo.indices()
    values_abs = torch.abs(A_coo.values())
    row_max.scatter_reduce_(0, rows, values_abs, reduce="amax", include_self=True)
    col_max.scatter_reduce_(0, cols, values_abs, reduce="amax", include_self=True)
    return row_max, col_max


def _row_col_absmax(A: Tensor) -> tuple[Tensor, Tensor]:
    if _is_sparse(A):
        return _sparse_row_col_absmax(A)
    return _amax_or_zero(A, dim=-1), _amax_or_zero(A, dim=-2)


def _safe_inv_sqrt(x: Tensor) -> Tensor:
    return torch.where(x > 0.0, torch.rsqrt(x), torch.ones_like(x))


def _cone_indices(cone: Cone, *, device: torch.device) -> Tensor:
    return cone.indices.to(device=device, dtype=torch.long)


def _rectify_cone_row_scaling(e: Tensor, cones: list[Cone]) -> Tensor:
    e = e.clone()
    for cone in cones:
        if isinstance(cone, ZeroCone | NonnegativeOrthant):
            continue
        if not isinstance(cone, SecondOrderCone | PositiveSemidefiniteCone):
            raise TypeError(f"Unsupported cone type for equilibration: {cone.__class__.__name__}")

        indices = _cone_indices(cone, device=e.device)
        if indices.numel() == 0:
            continue
        log_mean = torch.mean(torch.log(e[indices]))
        e[indices] = torch.exp(log_mean)
    return e


def _scale_dense_matrix_(A: Tensor, row_scale: Tensor, col_scale: Tensor) -> None:
    A.mul_(row_scale.unsqueeze(-1))
    A.mul_(col_scale.unsqueeze(-2))


def _scale_sparse_matrix_(A: Tensor, row_scale: Tensor, col_scale: Tensor) -> None:
    if A.layout == torch.sparse_coo:
        if not A.is_coalesced():
            raise ValueError("Sparse COO equilibration requires coalesced matrices for in-place scaling.")
        rows, cols = A.indices()
    elif A.layout == torch.sparse_csr:
        cols = A.col_indices()
        row_counts = A.crow_indices()[1:] - A.crow_indices()[:-1]
        rows = torch.repeat_interleave(torch.arange(A.shape[0], device=A.device), row_counts)
    elif A.layout == torch.sparse_csc:
        rows = A.row_indices()
        col_counts = A.ccol_indices()[1:] - A.ccol_indices()[:-1]
        cols = torch.repeat_interleave(torch.arange(A.shape[1], device=A.device), col_counts)
    else:
        raise TypeError(f"Unsupported sparse layout: {A.layout}")

    A.values().mul_(row_scale[rows] * col_scale[cols])


def _scale_matrix_(A: Tensor, row_scale: Tensor, col_scale: Tensor) -> None:
    if _is_sparse(A):
        _scale_sparse_matrix_(A, row_scale, col_scale)
    else:
        _scale_dense_matrix_(A, row_scale, col_scale)


def _cost_normalization(P: Tensor, q: Tensor) -> Tensor:
    _, p_col_norms = _row_col_absmax(P)
    p_mean_col_norm = torch.mean(p_col_norms)
    q_norm = torch.linalg.vector_norm(q, ord=float("inf"))
    denominator = torch.maximum(p_mean_col_norm, q_norm)
    return torch.where(denominator > 0.0, torch.reciprocal(denominator), torch.ones_like(denominator))


def equilibriate_(
    P: Float[Tensor, "n n"],
    q: Float[Tensor, " n"],
    A: Float[Tensor, "m n"],
    b: Float[Tensor, " m"],
    cones: list[Cone],
    max_iter: int = 10,
    conv_tol: float = 0.0,
) -> tuple[Float[Tensor, " n"], Float[Tensor, " m"], Float[Tensor, ""]]:
    """Equilibriate the problem data in-place using Ruiz scaling.

    The scaled data satisfies `P <- c D P D`, `q <- c D q`, `A <- E A D`, and `b <- E b`.
    Row scalings for non-separable cones are rectified to a single geometric-mean value per cone.
    """
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative.")
    if conv_tol < 0.0:
        raise ValueError("conv_tol must be nonnegative.")
    if P.dim() != 2 or A.dim() != 2 or q.dim() != 1 or b.dim() != 1:
        raise ValueError("Equilibration takes one problem at a time: P and A must be 2D. q and b must be 1D.")
    if P.shape[-1] != P.shape[-2]:
        raise ValueError("P must be square.")
    if A.shape[-1] != P.shape[-1] or q.shape[-1] != P.shape[-1] or b.shape[-1] != A.shape[-2]:
        raise ValueError("Incompatible P, q, A, and b shapes.")

    n = P.shape[-1]
    m = A.shape[-2]
    d = q.new_ones(n)
    e = b.new_ones(m)
    c = q.new_ones(())

    for _ in range(max_iter):
        _, p_col_norms = _row_col_absmax(P)
        a_row_norms, a_col_norms = _row_col_absmax(A)
        d_step = _safe_inv_sqrt(torch.maximum(p_col_norms, a_col_norms))
        e_step = _rectify_cone_row_scaling(_safe_inv_sqrt(a_row_norms), cones)

        _scale_matrix_(P, d_step, d_step)
        q.mul_(d_step)
        _scale_matrix_(A, e_step, d_step)
        b.mul_(e_step)

        gamma = _cost_normalization(P, q)
        P.mul_(gamma)
        q.mul_(gamma)

        d.mul_(d_step)
        e.mul_(e_step)
        c.mul_(gamma)

        step = torch.cat([d_step, e_step], dim=-1)
        if torch.max(torch.abs(1.0 - step)) <= conv_tol:
            break

    return d, e, c
