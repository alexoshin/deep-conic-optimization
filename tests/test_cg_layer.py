"""Matrix-free backward pass for the CG linear solve."""

import pytest
import torch

from dco.cg import LinearOperator, conjugate_gradient, jacobi_preconditioner
from dco.linsys import AdjointWarmStartCache, ReducedSystemOperator, solve_reduced_system
from dco.policy import HyperparameterPolicy
from dco.solver import ProjectionProx, Solver
from tests.helpers import random_qp

TIGHT = dict(max_iter=2000, eps_abs=0.0, eps_rel=1e-13)


def system(n=12, m=8, seed=0, sparse=True):
    g = torch.Generator().manual_seed(seed)
    P = torch.diag(torch.rand(n, generator=g) * 10)
    A = torch.randn(m, n, generator=g) * (torch.rand(m, n, generator=g) < 0.5)
    rhos = torch.rand(m, generator=g) + 0.1
    rhs = torch.randn(n, generator=g)
    sigma = 1e-6
    if sparse:
        P, A = P.to_sparse_csr(), A.to_sparse_csr()
    return P, A, rhos, rhs, sigma


def dense_matrix(P, A, rhos, sigma):
    P, A = P.to_dense(), A.to_dense()
    return P + sigma * torch.eye(P.shape[0]) + A.T @ torch.diag(rhos) @ A


def test_cg_solves_batched_spd():
    g = torch.Generator().manual_seed(1)
    L = torch.randn(3, 7, 7, generator=g)
    M = L @ L.mT + torch.eye(7)
    b = torch.randn(3, 7, generator=g)
    x, _ = conjugate_gradient(
        LinearOperator(M), b, preconditioner=jacobi_preconditioner(torch.diagonal(M, dim1=-2, dim2=-1)), **TIGHT
    )
    assert torch.allclose(x, torch.linalg.solve(M, b), atol=1e-9)


def test_implicit_gradients_match_autograd():
    """Check that `dL/drho` and `dL/drhs` match `torch.linalg.solve` on a dense matrix."""
    P, A, rhos, rhs, sigma = system()
    w = torch.randn(rhs.shape[0], generator=torch.Generator().manual_seed(9))

    rhos_a, rhs_a = rhos.clone().requires_grad_(True), rhs.clone().requires_grad_(True)
    linop = ReducedSystemOperator(P, A, rhos_a.detach(), sigma=sigma)
    x, _ = solve_reduced_system(linop, rhos_a, rhs_a, preconditioner=jacobi_preconditioner(linop.diagonal()), **TIGHT)
    (x * w).sum().backward()

    rhos_b, rhs_b = rhos.clone().requires_grad_(True), rhs.clone().requires_grad_(True)
    x_dense = torch.linalg.solve(dense_matrix(P, A, rhos_b, sigma), rhs_b)
    (x_dense * w).sum().backward()

    assert torch.allclose(x, x_dense, atol=1e-9)
    assert torch.allclose(rhos_a.grad, rhos_b.grad, atol=1e-8, rtol=1e-6)
    assert torch.allclose(rhs_a.grad, rhs_b.grad, atol=1e-8, rtol=1e-6)


def test_grad_rho_correctness():
    """Check `dL/drho = (A lambda) * (A x)` against the analytic adjoint."""
    P, A, rhos, rhs, sigma = system(sparse=False)
    w = torch.randn(rhs.shape[0], generator=torch.Generator().manual_seed(10))
    M = dense_matrix(P, A, rhos, sigma)
    x = torch.linalg.solve(M, rhs)
    lam = torch.linalg.solve(M, -w)
    expected = (A @ lam) * (A @ x)

    rhos_a = rhos.clone().requires_grad_(True)
    linop = ReducedSystemOperator(P, A, rhos_a.detach(), sigma=sigma)
    x_cg, _ = solve_reduced_system(linop, rhos_a, rhs, **TIGHT)
    (x_cg * w).sum().backward()
    assert torch.allclose(rhos_a.grad, expected, atol=1e-8, rtol=1e-6)


def test_grad_rho_matches_finite_differences():
    """Check `dL/drho` matches finite differences."""
    P, A, rhos, rhs, sigma = system(sparse=False)
    w = torch.randn(rhs.shape[0], generator=torch.Generator().manual_seed(11))
    v = torch.randn(rhos.shape[0], generator=torch.Generator().manual_seed(12))

    def loss(r):
        return (torch.linalg.solve(dense_matrix(P, A, r, sigma), rhs) * w).sum()

    h = 1e-6
    fd = (loss(rhos + h * v) - loss(rhos - h * v)) / (2 * h)
    rhos_a = rhos.clone().requires_grad_(True)
    x_cg, _ = solve_reduced_system(ReducedSystemOperator(P, A, rhos_a.detach(), sigma=sigma), rhos_a, rhs, **TIGHT)
    (x_cg * w).sum().backward()
    assert abs((rhos_a.grad * v).sum() - fd) <= 1e-6 * max(1.0, abs(fd))


def test_first_admm_block_paths():
    """The first ADMM block depends on `rho` through both the matrix (custom rule) AND the right-hand side (autograd).

    The first-block indirect gradients with respect to `rho` should equal the direct gradients.
    """
    P, q, A, b, cones = random_qp(10, 8, seed=3)
    g = torch.Generator().manual_seed(13)
    x, z, y = torch.randn(10, generator=g), torch.randn(8, generator=g), torch.randn(8, generator=g)
    rhos0 = torch.rand(8, generator=g) + 0.5
    wx, wz = torch.randn(10, generator=g), torch.randn(8, generator=g)
    solver = Solver(P, q, A, b, ProjectionProx(cones), policy=HyperparameterPolicy())

    rhos_ind = rhos0.clone().requires_grad_(True)
    linop = ReducedSystemOperator(
        solver.P, solver.A, rhos_ind.detach(), sigma=solver.sigma, A_T=solver.problem.A_T, A_sq_T=solver.problem.A_sq_T
    )
    x_t, z_t, _ = solver.first_block_update_indirect(
        linop,
        x,
        z,
        y,
        rhos_ind,
        x.clone(),
        2000,
        0.0,
        1e-13,
        False,
        "jacobi",
        None,
        None,
        None,
        AdjointWarmStartCache(),
        1,
    )
    ((x_t * wx).sum() + (z_t * wz).sum()).backward()

    rhos_dir = rhos0.clone().requires_grad_(True)
    coeff = solver.construct_coefficient_matrix(rhos_dir)
    from dco.functional import kkt_rhs, recover_z_tilde

    xnu = torch.linalg.solve(coeff, kkt_rhs(solver.sigma, x, solver.q, solver.b, z, y, rhos_dir))
    x_d, z_d = xnu[: solver.n], recover_z_tilde(z, xnu[solver.n :], y, rhos_dir)
    ((x_d * wx).sum() + (z_d * wz).sum()).backward()

    assert torch.allclose(x_t, x_d, atol=1e-9) and torch.allclose(z_t, z_d, atol=1e-9)
    assert torch.allclose(rhos_ind.grad, rhos_dir.grad, atol=1e-7, rtol=1e-6)


def test_implicit_gradient_matches_unrolling():
    """Check that the implicit layer gives the same gradients as differentiating the CG iterations."""
    P, A, rhos, rhs, sigma = system(sparse=False)
    w = torch.randn(rhs.shape[0], generator=torch.Generator().manual_seed(14))
    rhos_a = rhos.clone().requires_grad_(True)
    x_a, _ = conjugate_gradient(ReducedSystemOperator(P, A, rhos_a, sigma=sigma), rhs, **TIGHT)
    (x_a * w).sum().backward()
    rhos_b = rhos.clone().requires_grad_(True)
    x_b, _ = solve_reduced_system(ReducedSystemOperator(P, A, rhos_b.detach(), sigma=sigma), rhos_b, rhs, **TIGHT)
    (x_b * w).sum().backward()
    assert torch.allclose(rhos_a.grad, rhos_b.grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("warm_start", [False, True])
def test_adjoint_warm_start_does_not_change_the_unrolled_gradient(warm_start):
    P, q, A, b, cones = random_qp(8, 6, seed=4)
    policy = HyperparameterPolicy()
    rhos_param = torch.full((6,), 0.7, requires_grad=True)

    class RhoPolicy(HyperparameterPolicy):
        def forward(self, x, z, y, residuals, k, state):
            return rhos_param, state.alpha, state

    solver = Solver(P, q, A, b, ProjectionProx(cones), policy=RhoPolicy())
    result = solver.solve(
        max_iter=6,
        check_termination=0,
        linsys_solve_method="indirect",
        cg_preconditioner="jacobi",
        cg_eps_abs=0.0,
        cg_eps_rel=1e-13,
        adjoint_warm_start=warm_start,
        adaptive_rho_tolerance=1.0,
    )
    result.prim_res_norm_hist[-1].backward()
    grad = rhos_param.grad.clone()
    rhos_param.grad = None
    result = Solver(P, q, A, b, ProjectionProx(cones), policy=RhoPolicy()).solve(
        max_iter=6,
        check_termination=0,
        linsys_solve_method="indirect",
        cg_preconditioner="jacobi",
        cg_eps_abs=0.0,
        cg_eps_rel=1e-13,
        adjoint_warm_start=False,
        adaptive_rho_tolerance=1.0,
    )
    result.prim_res_norm_hist[-1].backward()
    assert torch.allclose(grad, rhos_param.grad, atol=1e-8, rtol=1e-6)
    del policy
