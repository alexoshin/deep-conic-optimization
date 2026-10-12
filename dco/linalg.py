"""Linear algebra functions."""

from jaxtyping import Bool, Float, Int
import numpy as np
import torch
from torch import Tensor

TRIL_INDICES_CACHE: dict[tuple[int, torch.device], Tensor] = {}
SVEC_SCALE_VECTOR_CACHE: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}


def tril_indices_cached(n: int, device: torch.device) -> Tensor:
    """Cached `torch.tril_indices(n, n)` per size and device."""
    key = (n, device)
    if key not in TRIL_INDICES_CACHE:
        TRIL_INDICES_CACHE[key] = torch.tril_indices(n, n, device=device)
    return TRIL_INDICES_CACHE[key]


def svec_scale_vector_cached(n: int, device: torch.device, dtype: torch.dtype) -> Float[Tensor, "n*(n+1)//2"]:
    """Cached `svec` scaling (1 on the diagonal, `sqrt(2)` off it)."""
    key = (n, device, dtype)
    if key not in SVEC_SCALE_VECTOR_CACHE:
        tril_inds = tril_indices_cached(n, device)
        diag_mask = tril_inds[0] == tril_inds[1]
        one = torch.as_tensor(1.0, dtype=dtype, device=device)
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        SVEC_SCALE_VECTOR_CACHE[key] = torch.where(diag_mask, one, sqrt2)
    return SVEC_SCALE_VECTOR_CACHE[key]


def svec(X: Float[Tensor, "... n n"]) -> Float[Tensor, "... n*(n+1)//2"]:
    """Symmetric vectorization of `X`: lower triangle, and off-diagonal elements scaled by `sqrt(2)`."""
    n = X.shape[-1]
    tril_inds = tril_indices_cached(n, X.device)
    scale_vec = svec_scale_vector_cached(n, X.device, X.dtype)
    return X[..., tril_inds[0], tril_inds[1]] * scale_vec


def smat(x: Float[Tensor, "... n*(n+1)//2"]) -> Float[Tensor, "... n n"]:
    """The symmetric matrix whose :func:`svec` is `x`."""
    *batch_dims, m = x.shape
    n = (int(np.sqrt(1 + 8 * m)) - 1) // 2
    X = torch.zeros(*batch_dims, n, n, dtype=x.dtype, device=x.device)
    tril_inds = tril_indices_cached(n, x.device)
    scale_vec = svec_scale_vector_cached(n, x.device, x.dtype)
    X[..., tril_inds[0], tril_inds[1]] = x / scale_vec
    return X + X.mT - torch.diag_embed(torch.diagonal(X, dim1=-2, dim2=-1))


def min_eigval_gap(X: Float[Tensor, "... n n"]) -> Float[Tensor, "..."]:
    """`min_{i != j} |lambda_i - lambda_j|` of a symmetric `X` (`nan` for `n < 2`)."""
    n = X.shape[-1]
    if n < 2:
        return torch.full(X.shape[:-2], float("nan"), dtype=X.dtype, device=X.device)
    with torch.no_grad():
        eigvals = torch.linalg.eigvalsh(0.5 * (X + X.mT))
        return (eigvals[..., 1:] - eigvals[..., :-1]).amin(dim=-1)


def dot_prod(x: Float[Tensor, "... n"], y: Float[Tensor, "... n"]) -> Float[Tensor, "..."]:
    """`x^T y` over the trailing dimension."""
    return torch.einsum("...n,...n->...", x, y)


def matvec(A: Float[Tensor, "... m n"], x: Float[Tensor, "... n"]) -> Float[Tensor, "... m"]:
    """`A x` for a sparse 2D `A` or a (batched) dense `A`."""
    if A.layout in (torch.sparse_csr, torch.sparse_csc, torch.sparse_coo):
        return torch.mv(A, x)
    return torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)


def L2_norm(x: Float[Tensor, "... n"]) -> Float[Tensor, "..."]:
    return torch.linalg.vector_norm(x, ord=2, dim=-1)


def inf_norm(x: Float[Tensor, "... n"]) -> Float[Tensor, "..."]:
    """Infinity norm (zero on an empty trailing dimension)."""
    if x.shape[-1] == 0:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    return torch.max(torch.abs(x), dim=-1).values


def lu_factor(A: Float[Tensor, "... n n"]) -> tuple[tuple[Tensor, Int[Tensor, "... n"]], Bool[Tensor, "..."]]:
    """Compute the LU factors of a batch of matrices along with per-matrix success flags."""
    LU, pivots, info = torch.linalg.lu_factor_ex(A)
    return (LU, pivots), info == 0


def lu_solve(LU_pivots: tuple[Tensor, Tensor], B: Float[Tensor, "... n k"]) -> Float[Tensor, "... n k"]:
    """Solve `A X = B` from the LU factors of `A`."""
    LU, pivots = LU_pivots
    return torch.linalg.lu_solve(LU, pivots, B)


def add_diag(P: Float[Tensor, "... n n"], diag: Float[Tensor, "... n"]) -> Float[Tensor, "... n n"]:
    """`P + diag(d)` for a dense (batched) or sparse (unbatched) `P`."""
    n = P.shape[-1]
    layout = P.layout
    if layout == torch.strided:
        return P + torch.diag_embed(diag)
    P = P.to_sparse_coo()
    diag_range = torch.arange(n, device=P.device)
    indices = torch.cat([P.indices(), torch.stack([diag_range, diag_range])], dim=1)
    values = torch.cat([P.values(), diag])
    P_new = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()
    if layout == torch.sparse_csr:
        return P_new.to_sparse_csr()
    if layout == torch.sparse_csc:
        return P_new.to_sparse_csc()
    return P_new


def safe_log(x: Tensor, eps: float = 1e-12) -> Tensor:
    return torch.log(x + eps)
