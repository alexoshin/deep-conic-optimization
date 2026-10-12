"""Block-diagonal super-batch representation of a conic program."""

from dataclasses import dataclass
import math

from jaxtyping import Float
import torch
from torch import Tensor

from .linalg import matvec
from .sparse import block_diag_csr, diagonal_csr
from .structure import MatrixStructure, detect_matrix_structure
from .super_batch import BatchMetadata, flat_super_matvec

# "single" reports per-problem scalars as scalars
# "uniform" splits flat super-vectors back into a `(B, ...)` grid
# "ragged" leaves them as flat vectors
SOURCES = ("single", "uniform", "ragged")


def uniform_batch_metadata(B: int, n: int, m: int, soc_strategy: str = "auto") -> BatchMetadata:
    """Super-batch layout of `B` same-shaped `(n, m)` sub-problems."""
    return BatchMetadata(B=B, n_per=n, m_per=m, soc_strategy=soc_strategy)


def apply_operator(M: Tensor, v: Float[Tensor, "... k"]) -> Float[Tensor, "... j"]:
    """Apply a (dense or sparse) 2D operator to a vector with optional leading dimensions."""
    if M.layout != torch.strided and v.dim() > 1:
        return flat_super_matvec(M, v)
    return matvec(M, v)


def transpose_operator(M: Tensor) -> Tensor:
    """Transpose of a 2D operator."""
    if M.layout == torch.strided:
        return M.mT
    return M.mT.to_sparse_csr()


def squared_transpose(M: Tensor) -> Tensor:
    """Returns the transpose of the entrywise square `(M * M)^T`.

    Used when computing `diag(A^T diag(rho) A)_i = sum_j rho_j A_ji^2`, such as in the Jacobi preconditioner.
    """
    if M.layout == torch.strided:
        return M.pow(2).mT
    csr = M if M.layout == torch.sparse_csr else M.to_sparse_csr()
    squared = torch.sparse_csr_tensor(csr.crow_indices(), csr.col_indices(), csr.values().pow(2), size=csr.shape)
    return squared.mT.to_sparse_csr()


def matrix_diagonal(M: Tensor) -> Float[Tensor, " n"]:
    """Compute the diagonal of a 2D (dense or sparse) square matrix."""
    if M.layout == torch.strided:
        return torch.diagonal(M, dim1=-2, dim2=-1)
    coo = M.to_sparse_coo().coalesce()
    indices, values = coo.indices(), coo.values()
    diag = torch.zeros(M.shape[-1], dtype=M.dtype, device=M.device)
    on_diagonal = indices[0] == indices[1]
    diag[indices[0][on_diagonal]] = values[on_diagonal]
    return diag


def empty_csr(n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Create an all-zero `(n, n)` CSR matrix with no stored entries."""
    return torch.sparse_csr_tensor(
        torch.zeros(n + 1, device=device, dtype=torch.int64),
        torch.empty(0, device=device, dtype=torch.int64),
        torch.empty(0, device=device, dtype=dtype),
        size=(n, n),
    )


def dense_stack_to_block_csr(M: Float[Tensor, "B r c"]) -> Tensor:
    """Stack a dense `(B, r, c)` block-diagonally into a single `(B r, B c)` CSR matrix.

    The sparsity pattern is the union of all nonzero positions.
    Entries missing in a particular block are stored as explicit zeros.
    """
    B, r, c = M.shape
    device = M.device
    mask = (M != 0).any(dim=0)
    if bool(mask.all()):
        crow = torch.arange(B * r + 1, device=device, dtype=torch.int64) * c
        columns = torch.arange(c, device=device, dtype=torch.int64).repeat(B * r)
        offsets = torch.arange(B, device=device, dtype=torch.int64).repeat_interleave(r * c) * c
        return torch.sparse_csr_tensor(crow, columns + offsets, M.reshape(-1), size=(B * r, B * c))
    rows, cols = mask.nonzero(as_tuple=True)
    nnz = rows.numel()
    counts = torch.bincount(rows, minlength=r)
    crow_block = torch.cat([torch.zeros(1, device=device, dtype=torch.int64), counts.cumsum(0)])
    block_ids = torch.arange(B, device=device, dtype=torch.int64)
    crow = torch.cat(
        [
            (block_ids.unsqueeze(1) * nnz + crow_block[:-1].unsqueeze(0)).reshape(-1),
            torch.tensor([B * nnz], device=device, dtype=torch.int64),
        ]
    )
    columns = (cols.unsqueeze(0) + block_ids.unsqueeze(1) * c).reshape(-1)
    values = M[:, rows, cols].reshape(-1)
    return torch.sparse_csr_tensor(crow, columns, values, size=(B * r, B * c))


@dataclass
class SuperProblem:
    """Block-diagonal super-batch of `B` conic programs.

    `A` is the `(total_m, total_n)` block-diagonal constraint matrix with its transpose and squared transpose precomputed.
    `P` is stored according to `P_structure`: nothing for `ZERO`, the diagonal for `DIAGONAL`, and the matrix itself for `SPARSE`.
    `q` and `b` are flat super-vectors.
    """

    q: Float[Tensor, " total_n"]
    A: Float[Tensor, "total_m total_n"]
    A_T: Float[Tensor, "total_n total_m"]
    A_sq_T: Float[Tensor, "total_n total_m"]
    b: Float[Tensor, " total_m"]
    metadata: BatchMetadata
    P_structure: MatrixStructure
    P: Tensor | None = None
    source: str = "single"

    @property
    def B(self) -> int:
        """Number of sub-problems."""
        return self.metadata.B

    @property
    def n(self) -> int:
        """Total number of variables."""
        return self.metadata.total_n

    @property
    def m(self) -> int:
        """Total number of constraint rows."""
        return self.metadata.total_m

    def reducers(self, device: torch.device):
        """Create `(reducer_n, reducer_m)` segment reducers for the flat super-vectors."""
        return self.metadata.reducers(device)

    def matvec_A(self, v: Float[Tensor, "... total_n"]) -> Float[Tensor, "... total_m"]:
        """Compute `A v`."""
        return apply_operator(self.A, v)

    def matvec_AT(self, v: Float[Tensor, "... total_m"]) -> Float[Tensor, "... total_n"]:
        """Compute `A^T v`."""
        return apply_operator(self.A_T, v)

    def matvec_P(self, v: Float[Tensor, "... total_n"]) -> Float[Tensor, "... total_n"]:
        """Compute `P v`."""
        if self.P_structure is MatrixStructure.ZERO:
            return torch.zeros_like(v)
        if self.P_structure is MatrixStructure.DIAGONAL:
            return self.P * v
        return apply_operator(self.P, v)

    def diag_A_sq_T(self, rhos: Float[Tensor, " total_m"]) -> Float[Tensor, " total_n"]:
        """Compute `diag(A^T diag(rho) A)`."""
        return apply_operator(self.A_sq_T, rhos)

    def P_matrix(self) -> Tensor:
        """Compute `P` in matrix form."""
        if self.P_structure is MatrixStructure.ZERO:
            return empty_csr(self.n, self.q.dtype, self.q.device)
        if self.P_structure is MatrixStructure.DIAGONAL:
            return diagonal_csr(self.P)
        return self.P

    def P_dense(self) -> Float[Tensor, "total_n total_n"]:
        """Return `P` as a dense matrix."""
        if self.P_structure is MatrixStructure.ZERO:
            return torch.zeros(self.n, self.n, dtype=self.q.dtype, device=self.q.device)
        if self.P_structure is MatrixStructure.DIAGONAL:
            return torch.diag_embed(self.P)
        return self.P.to_dense() if self.P.layout != torch.strided else self.P

    def A_dense(self) -> Float[Tensor, "total_m total_n"]:
        """Return `A` as a dense matrix."""
        return self.A.to_dense() if self.A.layout != torch.strided else self.A

    @classmethod
    def assemble(
        cls,
        P: Tensor | None,
        q: Float[Tensor, " total_n"],
        A: Tensor,
        b: Float[Tensor, " total_m"],
        metadata: BatchMetadata,
        structure_dispatch: bool,
        source: str,
        structure: MatrixStructure | None = None,
    ) -> "SuperProblem":
        """Assemble the super-problem."""
        if structure is None:
            structure = detect_matrix_structure(P) if structure_dispatch else MatrixStructure.SPARSE
        if structure is MatrixStructure.ZERO:
            stored_P = None
        elif structure is MatrixStructure.DIAGONAL:
            stored_P = matrix_diagonal(P)
        else:
            stored_P = P
        return cls(
            q=q,
            A=A,
            A_T=transpose_operator(A),
            A_sq_T=squared_transpose(A),
            b=b,
            metadata=metadata,
            P_structure=structure,
            P=stored_P,
            source=source,
        )

    @classmethod
    def from_single(
        cls,
        P: Float[Tensor, "n n"],
        q: Float[Tensor, " n"],
        A: Float[Tensor, "m n"],
        b: Float[Tensor, " m"],
        structure_dispatch: bool = True,
        soc_strategy: str = "auto",
    ) -> "SuperProblem":
        """Assemble a single problem as a super-batch with `B = 1`."""
        m, n = A.shape
        metadata = BatchMetadata(B=1, n_per=n, m_per=m, soc_strategy=soc_strategy)
        return cls.assemble(P, q, A, b, metadata, structure_dispatch, "single")

    @classmethod
    def from_uniform(
        cls,
        P: Float[Tensor, "... n n"],
        q: Float[Tensor, "... n"],
        A: Float[Tensor, "... m n"],
        b: Float[Tensor, "... m"],
        structure_dispatch: bool = True,
        soc_strategy: str = "auto",
    ) -> "SuperProblem":
        """Create a dense batch of same-shaped problems as a uniform super-batch."""
        m, n = A.shape[-2:]
        B = math.prod(A.shape[:-2])
        P_batch = P.reshape(B, n, n)
        structure = detect_matrix_structure(P_batch) if structure_dispatch else MatrixStructure.SPARSE
        P_super = None if structure is MatrixStructure.ZERO else dense_stack_to_block_csr(P_batch)
        metadata = BatchMetadata(B=B, n_per=n, m_per=m, soc_strategy=soc_strategy)
        return cls.assemble(
            P_super,
            q.reshape(B * n),
            dense_stack_to_block_csr(A.reshape(B, m, n)),
            b.reshape(B * m),
            metadata,
            structure_dispatch,
            "uniform",
            structure=structure,
        )

    @classmethod
    def from_ragged(
        cls,
        problems: list[tuple[Tensor, Tensor, Tensor, Tensor]],
        soc_strategy: str = "auto",
        structure_dispatch: bool = True,
    ) -> "SuperProblem":
        """Stack `(P, q, A, b)` sub-problems of different sizes into a single block-diagonal problem."""
        if not problems:
            raise ValueError("from_ragged requires at least one sub-problem")
        Ps, qs, As, bs = zip(*problems)
        metadata = BatchMetadata(
            B=len(problems),
            n_pers=[q.shape[-1] for q in qs],
            m_pers=[b.shape[-1] for b in bs],
            soc_strategy=soc_strategy,
        )
        return cls.assemble(
            block_diag_csr(*Ps),
            torch.cat(list(qs)),
            block_diag_csr(*As),
            torch.cat(list(bs)),
            metadata,
            structure_dispatch,
            "ragged",
        )

    @classmethod
    def from_super_batch(
        cls,
        P: Tensor,
        q: Float[Tensor, " total_n"],
        A: Tensor,
        b: Float[Tensor, " total_m"],
        metadata: BatchMetadata,
        structure_dispatch: bool = True,
    ) -> "SuperProblem":
        """Wrap data that is already stacked block-diagonally (e.g. from a dataset sampler)."""
        return cls.assemble(P, q, A, b, metadata, structure_dispatch, "ragged")

    @classmethod
    def build(
        cls,
        P: Tensor,
        q: Tensor,
        A: Tensor,
        b: Tensor,
        metadata: BatchMetadata | None = None,
        structure_dispatch: bool = True,
    ) -> "SuperProblem":
        """Build the super-problem matching the layout of the data."""
        if A.dim() > 2:
            m, n = A.shape[-2:]
            B = math.prod(A.shape[:-2])
            if metadata is not None:
                expected = uniform_batch_metadata(B, n, m, metadata.soc_strategy)
                if (metadata.B, metadata.n_pers, metadata.m_pers) != (B, expected.n_pers, expected.m_pers):
                    raise ValueError(
                        f"batch_metadata describes B={metadata.B} sub-problems of sizes "
                        f"n_pers={metadata.n_pers}, m_pers={metadata.m_pers}, but the dense batch is "
                        f"B={B} problems of size n={n}, m={m}."
                    )
            soc_strategy = "auto" if metadata is None else metadata.soc_strategy
            return cls.from_uniform(P, q, A, b, structure_dispatch, soc_strategy=soc_strategy)
        if metadata is not None:
            return cls.from_super_batch(P, q, A, b, metadata, structure_dispatch)
        return cls.from_single(P, q, A, b, structure_dispatch)
