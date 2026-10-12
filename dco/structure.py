"""Sparsity structure detection for the first-block linear system."""

from enum import Enum

from jaxtyping import Float
import torch
from torch import Tensor


class MatrixStructure(Enum):
    """Structure of the quadratic term `P`."""

    ZERO = "zero"
    DIAGONAL = "diagonal"
    SPARSE = "sparse"


def detect_matrix_structure(P: Float[Tensor, "... n n"]) -> MatrixStructure:
    """Identify `P` as `ZERO` (no nonzero entry), `DIAGONAL` (no nonzero off-diagonal entry), or `SPARSE`.

    Handles dense (optionally batched) and sparse (COO / CSR / CSC, unbatched) matrices.
    """
    if P.layout == torch.strided:
        nnz = int(torch.count_nonzero(P))
        if nnz == 0:
            return MatrixStructure.ZERO
        nnz_diag = int(torch.count_nonzero(torch.diagonal(P, dim1=-2, dim2=-1)))
        return MatrixStructure.DIAGONAL if nnz == nnz_diag else MatrixStructure.SPARSE

    P_coo = P.to_sparse_coo().coalesce()
    indices, values = P_coo.indices(), P_coo.values()
    nonzero = values != 0.0
    if not bool(nonzero.any()):
        return MatrixStructure.ZERO
    off_diagonal = nonzero & (indices[0] != indices[1])
    return MatrixStructure.SPARSE if bool(off_diagonal.any()) else MatrixStructure.DIAGONAL


def max_nnz_per_row(A: Float[Tensor, "... m n"]) -> int:
    """Largest number of nonzeros in any row of `A`.

    Sparse layouts are counted by stored entries,
    so explicitly stored zeros can only increase the count.
    The result never undercounts the actual number of nonzeros.
    """
    if A.shape[-2] == 0:
        return 0
    if A.layout == torch.strided:
        return int((A != 0.0).sum(dim=-1).max())
    if A.layout == torch.sparse_csr:
        crow = A.crow_indices()
        return int((crow[1:] - crow[:-1]).max())
    rows = A.to_sparse_coo().coalesce().indices()[0]
    return int(torch.bincount(rows).max()) if rows.numel() > 0 else 0
