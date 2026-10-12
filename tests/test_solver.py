"""COSMO solver."""

import torch

from dco.cones import ZeroCone
from dco.loss import compute_loss
from dco.policy import AdaptiveRhoOSQP, FeedbackPolicy, HyperparameterPolicy
from dco.problem import uniform_batch_metadata
from dco.solver import ProjectionProx, Solver, Solver_Status
from dco.super_batch import BatchMetadata
from dco.sparse import block_diag_csr
from dco.warmstart import RowWiseMLPWarmstart
from tests.helpers import mixed_cone_problem, random_qp


SETTINGS = dict(
    max_iter=5000,
    eps_abs=1e-8,
    eps_rel=0.0,
    check_termination=1,
    cg_eps_abs=0.0,
    cg_eps_rel=1e-12,
    cg_preconditioner="jacobi",
)

# Fixed-length runs for comparing iterate trajectories
TRAJECTORY_SETTINGS = dict(
    check_termination=0,
    cg_eps_abs=0.0,
    cg_eps_rel=1e-12,
    cg_preconditioner="jacobi",
    adaptive_rho_tolerance=1.0,
)


def test_equality_constrained_qp():
    g = torch.Generator().manual_seed(0)
    n = 8
    m = 4

    L = torch.randn(n, n, generator=g)
    P = L @ L.T + torch.eye(n)
    q = torch.randn(n, generator=g)
    A = torch.randn(m, n, generator=g)
    b = torch.randn(m, generator=g)
    cones = [ZeroCone(range(m))]
    prox = ProjectionProx(cones)

    # Analytic solution
    kkt = torch.zeros(n + m, n + m)
    kkt[:n, :n], kkt[:n, n:], kkt[n:, :n] = P, A.T, A
    x_star = torch.linalg.solve(kkt, torch.cat([-q, b]))[:n]

    for method in ("direct", "indirect"):
        solver = Solver(P, q, A, b, prox, policy=AdaptiveRhoOSQP())
        result = solver.solve(linsys_solve_method=method, **SETTINGS)
        assert result.status is Solver_Status.CONVERGED
        assert torch.allclose(result.x, x_star, atol=1e-6)


def test_direct_indirect_follow_same_trajectory():
    P, q, A, b, cones = mixed_cone_problem()
    prox = ProjectionProx(cones)

    results = {}
    for method in ("direct", "indirect"):
        solver = Solver(P, q, A, b, prox, policy=AdaptiveRhoOSQP())
        results[method] = solver.solve(max_iter=40, linsys_solve_method=method, **TRAJECTORY_SETTINGS)

    assert torch.allclose(results["direct"].x_hist, results["indirect"].x_hist, atol=1e-8)
    assert torch.allclose(results["direct"].y_hist, results["indirect"].y_hist, atol=1e-8)


def test_mixed_cone_problem_converges():
    P, q, A, b, cones = mixed_cone_problem()
    prox = ProjectionProx(cones)

    solver = Solver(P, q, A, b, prox, policy=AdaptiveRhoOSQP())
    result = solver.solve(linsys_solve_method="indirect", **SETTINGS)
    assert result.status is Solver_Status.CONVERGED

    # KKT conditions
    x, y = result.x, result.y
    s = b - A @ x
    assert (P @ x + q + A.T @ y).abs().max() < 1e-6  # stationarity
    assert prox.violation(s).abs().max() < 1e-6  # primal feasibility
    assert (-prox.project(-y)).abs().max() < 1e-6  # dual feasibility: y in K^*
    assert abs(s @ y) < 1e-5  # complementarity


def test_dense_batch_matches_sequential_solves():
    g = torch.Generator().manual_seed(0)
    B = 3

    P, q, A, b, cones = mixed_cone_problem()
    n, m = q.shape[0], b.shape[0]
    qs = q + 0.1 * torch.randn(B, n, generator=g)
    prox = ProjectionProx(cones, uniform_batch_metadata(B, n, m))

    solver = Solver(P.expand(B, -1, -1), qs, A.expand(B, -1, -1), b.expand(B, -1), prox, policy=AdaptiveRhoOSQP())
    batched = solver.solve(max_iter=30, linsys_solve_method="indirect", **TRAJECTORY_SETTINGS)
    x_hist = batched.x_hist.reshape(-1, B, n)

    for i in range(B):
        solver = Solver(P, qs[i], A, b, ProjectionProx(cones), policy=AdaptiveRhoOSQP())
        single = solver.solve(max_iter=30, linsys_solve_method="indirect", **TRAJECTORY_SETTINGS)
        assert torch.allclose(x_hist[:, i], single.x_hist, atol=1e-8)


def test_ragged_super_batch_matches_sequential_solves():
    problems = [random_qp(6, 4, seed=0), random_qp(9, 6, seed=1)]

    # Stack the problems block-diagonally
    P = block_diag_csr(*(p[0].to_sparse_csr() for p in problems))
    q = torch.cat([p[1] for p in problems])
    A = block_diag_csr(*(p[2].to_sparse_csr() for p in problems))
    b = torch.cat([p[3] for p in problems])
    metadata = BatchMetadata(B=2, n_pers=[6, 9], m_pers=[4, 6])
    prox = ProjectionProx([p[4] for p in problems], batch_metadata=metadata)

    solver = Solver(P, q, A, b, prox, policy=AdaptiveRhoOSQP(), batch_metadata=metadata)
    result = solver.solve(max_iter=25, linsys_solve_method="indirect", **TRAJECTORY_SETTINGS)

    offset = 0
    for Pi, qi, Ai, bi, cones in problems:
        solver = Solver(Pi, qi, Ai, bi, ProjectionProx(cones), policy=AdaptiveRhoOSQP())
        single = solver.solve(max_iter=25, linsys_solve_method="indirect", **TRAJECTORY_SETTINGS)
        assert torch.allclose(result.x_hist[..., offset : offset + qi.shape[0]], single.x_hist, atol=1e-8)
        offset += qi.shape[0]


def test_untrained_policy_reproduces_fixed_hyperparameter_cosmo():
    P, q, A, b, cones = mixed_cone_problem()
    prox = ProjectionProx(cones)

    results = {}
    for name, policy in (("learned", FeedbackPolicy(rectify_rhos_per_cone=True)), ("fixed", HyperparameterPolicy())):
        solver = Solver(P, q, A, b, prox, policy=policy)
        results[name] = solver.solve(max_iter=20, linsys_solve_method="indirect", **TRAJECTORY_SETTINGS)

    assert torch.allclose(results["learned"].x_hist, results["fixed"].x_hist, atol=1e-8)
    assert torch.allclose(results["learned"].rhos_hist, results["fixed"].rhos_hist)


def test_rectification():
    P, q, A, b, cones = mixed_cone_problem()
    m = b.shape[0]

    policy = FeedbackPolicy(rectify_rhos_per_cone=True)
    policy.initialize(P, q, A, b, cones)
    rhos = policy.rectify_rhos(torch.rand(m) + 0.1)

    assert torch.allclose(rhos[5:9], rhos[5:6].expand(4))  # SOC block
    assert torch.allclose(rhos[9:], rhos[9:10].expand(6))  # PSD block


def test_policy_warmstart_gradients():
    g = torch.Generator().manual_seed(0)
    B = 2
    k = 15

    P, q, A, b, cones = mixed_cone_problem()
    n, m = q.shape[0], b.shape[0]
    sample = (
        P.expand(B, -1, -1),
        q + 0.1 * torch.randn(B, n, generator=g),
        A.expand(B, -1, -1),
        b.expand(B, -1),
        cones,
        torch.zeros(B, n),
        torch.zeros(B, m),
        torch.zeros(B, m),
    )

    policy = FeedbackPolicy(rectify_rhos_per_cone=True)
    init = RowWiseMLPWarmstart()
    loss, result, gap = compute_loss(sample, policy, k, initialization=init)
    loss.backward()
    assert torch.isfinite(loss)
    assert gap.shape == (B, k + 1)

    # Check gradients reach both networks
    params = list(policy.parameters()) + list(init.parameters())
    assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    assert policy.rho_policy.mlp[-1].weight.grad.abs().sum() > 0
    assert init.x_net.mlp[-1].weight.grad.abs().sum() > 0
