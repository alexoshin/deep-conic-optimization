"""PSD projection backward rule."""

import pytest
import torch

from dco.cones import BatchedPositiveSemidefiniteCone, PositiveSemidefiniteCone, psd_project
from dco.linalg import smat, svec
from tests.helpers import clustered_symmetric


def autograd_projection(S: torch.Tensor) -> torch.Tensor:
    """Compute the projection through `eigh` (uses autograd in the backward pass)."""
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (S + S.mT))
    return eigvecs @ torch.diag_embed(torch.clamp(eigvals, min=0.0)) @ eigvecs.mT


def symmetric(n: int, generator: torch.Generator) -> torch.Tensor:
    G = torch.randn(n, n, generator=generator)
    return 0.5 * (G + G.T)


def vjp(projection, S: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    S = S.clone().requires_grad_(True)
    (projection(S) * G).sum().backward()
    return S.grad


@pytest.mark.parametrize("n", [4, 10])
def test_dk_matches_autograd_well_separated(n):
    g = torch.Generator().manual_seed(1)
    S, G = symmetric(n, g), symmetric(n, g)
    assert torch.allclose(vjp(psd_project, S, G), vjp(autograd_projection, S, G), atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("gap", [1e-2, 1e-6, 1e-10])
def test_dk_matches_finite_differences(gap):
    n = 10
    g = torch.Generator().manual_seed(2)
    S = clustered_symmetric(n, gap, seed=3)
    G, V = symmetric(n, g), symmetric(n, g)
    h = 1e-7 * min(1.0, gap / 1e-3) if gap < 1e-3 else 1e-7
    fd = ((psd_project(S + h * V) - psd_project(S - h * V)) * G).sum() / (2 * h)
    dk = (vjp(psd_project, S, G) * V).sum()
    assert torch.isfinite(dk)
    assert abs(dk - fd) <= 1e-5 * max(1.0, abs(fd))


def test_dk_finite_at_repeated_eigenvalues():
    """On `S = c I` (every eigenvalue equal) the projection is the identity, so its VJP is `sym(G)`."""
    n = 6
    g = torch.Generator().manual_seed(4)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    S = 2.0 * Q @ Q.T
    G = symmetric(n, g)
    grad_dk = vjp(psd_project, S, G)
    assert torch.isfinite(grad_dk).all()
    assert torch.allclose(grad_dk, 0.5 * (G + G.T), atol=1e-8)
    grad_autograd = vjp(autograd_projection, S, G)
    assert not torch.isfinite(grad_autograd).all() or (grad_autograd - grad_dk).abs().max() > 1e-3


def test_dk_matches_autograd_mixed_sign():
    """Check on a matrix with eigenvalues `(-1, -1, 0.5, 2)`."""
    g = torch.Generator().manual_seed(5)
    Q, _ = torch.linalg.qr(torch.randn(4, 4, generator=g))
    lam = torch.tensor([-1.0, -1.0, 0.5, 2.0])
    S = Q @ torch.diag(lam) @ Q.T
    G = symmetric(4, g)
    f = torch.clamp(lam, min=0.0)
    L = torch.zeros(4, 4)
    for i in range(4):
        for j in range(4):
            L[i, j] = (f[j] - f[i]) / (lam[j] - lam[i]) if lam[i] != lam[j] else float(lam[i] > 0)
    Gs = 0.5 * (G + G.T)
    expected = Q @ (L * (Q.T @ Gs @ Q)) @ Q.T
    grad_dk = vjp(psd_project, S, G)
    assert torch.allclose(grad_dk, expected, atol=1e-8)


def test_projection_forward():
    g = torch.Generator().manual_seed(6)
    S = symmetric(5, g)
    z = svec(S)
    assert torch.allclose(
        PositiveSemidefiniteCone(range(15), "dk")(z), PositiveSemidefiniteCone(range(15), "autograd")(z)
    )
    out = smat(PositiveSemidefiniteCone(range(15))(z))
    assert torch.linalg.eigvalsh(out).min() >= -1e-12


def test_batched_psd_cone_forward():
    g = torch.Generator().manual_seed(7)
    k = 6  # svec of 3x3
    z = torch.randn(3 * k, generator=g)
    cones = [PositiveSemidefiniteCone(range(i * k, (i + 1) * k)) for i in range(3)]
    expected = z
    for cone in cones:
        expected = cone(expected)
    batched = BatchedPositiveSemidefiniteCone(torch.arange(3 * k).reshape(3, k))
    assert torch.allclose(batched(z), expected)
