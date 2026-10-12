"""Sparse matrix helpers (CSR construction and block-diagonal stacking)."""

from jaxtyping import Float
import torch
from torch import Tensor


def convert(A: Tensor, layout: str) -> Tensor:
    """Convert a tensor to the specified layout."""
    if layout == "csr":
        return A.to_sparse_csr()
    if layout == "csc":
        return A.to_sparse_csc()
    if layout == "coo":
        return A.to_sparse_coo()
    if layout == "dense":
        return A.to_dense()
    raise ValueError(f"Unknown layout: {layout}")


def random_matrix(m: int, n: int, sparsity: float = 0.2, layout: str = "csr") -> Float[Tensor, "m n"]:
    """Generate a random `(m, n)` matrix with `sparsity * m * n` normally distributed entries."""
    nnz = int(m * n * sparsity)
    indices = torch.stack([torch.randint(0, m, (nnz,)), torch.randint(0, n, (nnz,))])
    A = torch.sparse_coo_tensor(indices, torch.randn(nnz), size=(m, n)).coalesce()
    return convert(A, layout)


def diagonal_csr(diag: Float[Tensor, " n"]) -> Tensor:
    """`diag(v)` as a CSR matrix on the values' device."""
    n = diag.numel()
    index = torch.arange(n + 1, device=diag.device, dtype=torch.int64)
    return torch.sparse_csr_tensor(index, index[:n], diag, size=(n, n))


def block_diag_csr(*Ms: Tensor) -> Tensor:
    """Block-diagonal stack of 2D sparse matrices of possibly different shapes, as CSR."""
    crow_chunks: list[Tensor] = [torch.tensor([0], device=Ms[0].device, dtype=torch.int64)]
    col_chunks: list[Tensor] = []
    value_chunks: list[Tensor] = []
    total_nnz = total_rows = total_cols = 0
    for M in Ms:
        M_csr = M if M.layout == torch.sparse_csr else M.to_sparse_csr()
        crow_chunks.append(M_csr.crow_indices()[1:].to(torch.int64) + total_nnz)
        col_chunks.append(M_csr.col_indices().to(torch.int64) + total_cols)
        value_chunks.append(M_csr.values())
        total_nnz += M_csr.values().numel()
        total_rows += M_csr.shape[0]
        total_cols += M_csr.shape[1]
    return torch.sparse_csr_tensor(
        torch.cat(crow_chunks),
        torch.cat(col_chunks),
        torch.cat(value_chunks),
        size=(total_rows, total_cols),
        dtype=Ms[0].dtype,
        device=Ms[0].device,
    )
