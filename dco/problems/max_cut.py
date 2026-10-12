"""Max-cut SDPs."""

from jaxtyping import Float
import torch
from torch import Tensor
from ..linalg import svec
from ..cones import Cone, ZeroCone, PositiveSemidefiniteCone


def generate_max_cut_sdp(
    n: int = 10,
    edge_density: float = 0.5,
    layout: str = "csr",
) -> tuple[
    Float[Tensor, "n n"],
    Float[Tensor, "n"],
    Float[Tensor, "m n"],
    Float[Tensor, "m"],
    list[Cone],
]:
    """Generate a random max-cut SDP."""

    # Sample an unweighted Erdos-Renyi graph G(n, p)
    A_graph = (torch.rand(n, n) < edge_density).to(torch.get_default_dtype())

    # Make symmetric and remove self-loops
    A_lower = torch.tril(A_graph, diagonal=-1)
    A_graph = A_lower + A_lower.T

    # Compute the Laplacian L = D - A
    L = torch.diag(torch.sum(A_graph, dim=1)) - A_graph

    # Generate objective data
    n_vars = n * (n + 1) // 2
    q = -svec(0.25 * L)

    # Diagonal constraints W_ii = 1
    tril_inds = torch.tril_indices(n, n)
    diag_inds = torch.where(tril_inds[0] == tril_inds[1])[0]
    b = torch.cat([torch.ones(n), torch.zeros(n_vars)])
    cones = [ZeroCone(torch.arange(n)), PositiveSemidefiniteCone(torch.arange(n, n + n_vars))]

    if layout == "dense":
        P = torch.zeros(n_vars, n_vars)
        A_diag = torch.zeros(n, n_vars)
        A_diag[torch.arange(n), diag_inds] = 1.0
        # SDP constraint: -X + S = 0, S >= 0
        A = torch.cat([A_diag, -torch.eye(n_vars)], dim=0)
        return P, q, A, b, cones

    # Sparse path: the dense path materializes O(n_vars^2) zeros (P is ~125 GB at n = 500)
    # P is identically zero and A has exactly n + n_vars nonzeros
    row = torch.cat([torch.arange(n), torch.arange(n, n + n_vars)])
    col = torch.cat([diag_inds, torch.arange(n_vars)])
    val = torch.cat([torch.ones(n), -torch.ones(n_vars)])
    A = torch.sparse_coo_tensor(torch.stack([row, col]), val, (n + n_vars, n_vars)).coalesce()
    P = torch.sparse_coo_tensor(
        torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=q.dtype), (n_vars, n_vars)
    ).coalesce()

    if layout == "csr":
        P, A = P.to_sparse_csr(), A.to_sparse_csr()
    elif layout == "csc":
        P, A = P.to_sparse_csc(), A.to_sparse_csc()
    # layout == "coo": already COO

    return P, q, A, b, cones
