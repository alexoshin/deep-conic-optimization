"""Lovasz theta SDPs."""

from jaxtyping import Float
import torch
from torch import Tensor
from ..linalg import svec
from ..cones import Cone, ZeroCone, PositiveSemidefiniteCone


def generate_lovasz_theta_sdp(
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
    """Generate a random Lovász theta function SDP."""

    # Sample an unweighted Erdos-Renyi graph G(n, p)
    A_graph = torch.rand(n, n) < edge_density

    # Make symmetric and remove self-loops
    A_lower = torch.tril(A_graph, diagonal=-1)
    A_graph = A_lower + A_lower.T

    # Generate objective data
    n_vars = n * (n + 1) // 2
    P = torch.zeros(n_vars, n_vars)
    q = -svec(torch.ones(n, n))

    # Generate equality constraints
    A_eqs = []
    b_eqs = []

    # Trace constraint
    a = torch.zeros(n_vars)
    tril_inds = torch.tril_indices(n, n)
    diag_mask = tril_inds[0] == tril_inds[1]
    diag_inds = torch.where(diag_mask)[0]
    a[diag_inds] = 1.0
    A_eqs.append(a)
    b_eqs.append(torch.tensor([1.0], dtype=torch.get_default_dtype()))

    # Adjacency constraints
    for i, (row, col) in enumerate(zip(tril_inds[0], tril_inds[1])):
        if A_graph[row, col]:
            a = torch.zeros(n_vars)
            a[i] = 1.0
            A_eqs.append(a)
            b_eqs.append(torch.tensor([0.0], dtype=torch.get_default_dtype()))

    A_eq = torch.stack(A_eqs)
    b_eq = torch.cat(b_eqs)

    # SDP constraint: X >= 0 <=> -X + S = 0, S >= 0
    m_sdp = n_vars
    G = -torch.eye(n_vars)
    h = torch.zeros(n_vars)

    # Combined constraints
    A = torch.cat([A_eq, G], dim=0)
    b = torch.cat([b_eq, h], dim=0)

    n_eq = b_eq.shape[0]
    n_total_constraints = n_eq + m_sdp
    cones = [
        ZeroCone(torch.arange(n_eq)),
        PositiveSemidefiniteCone(torch.arange(n_eq, n_total_constraints)),
    ]

    if layout == "coo":
        P = P.to_sparse_coo()
        A = A.to_sparse_coo()
    elif layout == "csr":
        P = P.to_sparse_csr()
        A = A.to_sparse_csr()
    elif layout == "csc":
        P = P.to_sparse_csc()
        A = A.to_sparse_csc()
    else:  # layout == "dense"
        pass

    return P, q, A, b, cones
