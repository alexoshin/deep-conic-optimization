"""Common problem builders."""

import torch

from dco.cones import NonnegativeOrthant, PositiveSemidefiniteCone, SecondOrderCone, ZeroCone
from dco.linalg import svec


def random_qp(n: int, m: int, seed: int = 0):
    """Generate a strongly convex QP with `m` constraint rows: half equalities and half inequalities."""
    g = torch.Generator().manual_seed(seed)
    L = torch.randn(n, n, generator=g)
    P = L @ L.T + 0.5 * torch.eye(n)
    q = torch.randn(n, generator=g)
    A = torch.randn(m, n, generator=g)
    b = torch.randn(m, generator=g)
    k = m // 2
    cones = [ZeroCone(range(0, k)), NonnegativeOrthant(range(k, m))]
    return P, q, A, b, cones


def mixed_cone_problem(n: int = 12, psd_size: int = 3, seed: int = 0):
    """Generate a problem with zero, nonnegative, second-order, and PSD blocks."""
    g = torch.Generator().manual_seed(seed)
    k_psd = psd_size * (psd_size + 1) // 2
    sizes = [2, 3, 4, k_psd]
    m = sum(sizes)
    L = torch.randn(n, n, generator=g)
    P = L @ L.T + 0.5 * torch.eye(n)
    q = torch.randn(n, generator=g)
    A = torch.randn(m, n, generator=g)
    # Ensure b - A x is feasible
    x0 = torch.randn(n, generator=g)
    s = torch.zeros(m)
    s[2:5] = torch.rand(3, generator=g) + 0.1
    soc = torch.randn(4, generator=g)
    soc[0] = soc[1:].norm() + 0.5
    s[5:9] = soc
    G = torch.randn(psd_size, psd_size, generator=g)
    s[9:] = svec(G @ G.T + 0.1 * torch.eye(psd_size))
    b = A @ x0 + s
    cones = [
        ZeroCone(range(0, 2)),
        NonnegativeOrthant(range(2, 5)),
        SecondOrderCone(range(5, 9)),
        PositiveSemidefiniteCone(range(9, m)),
    ]
    return P, q, A, b, cones


def clustered_symmetric(n: int, gap: float, seed: int = 0) -> torch.Tensor:
    """Generate a symmetric matrix whose eigenvalues sit in two clusters at `-gap/2` and `+gap/2` with spread `gap * 1e-3`."""
    g = torch.Generator().manual_seed(seed)
    half = n // 2
    spread = gap * 1e-3
    lam = torch.cat(
        [
            -gap / 2 + spread * (2 * torch.rand(half, generator=g) - 1),
            gap / 2 + spread * (2 * torch.rand(n - half, generator=g) - 1),
        ]
    )
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    return Q @ torch.diag(lam) @ Q.T
