"""Clarabel interface.

Solves a cone problem in the standard form using Clarabel.
The data (P, q, A, b, K) already matches the form that Clarabel expects.
The dual variable sign and the `svec` ordering (column-major upper triangle, off-diagonals scaled by `sqrt(2)`)
matches Clarabel's conventions directly.
"""

import clarabel as clarabel_native
from jaxtyping import Float
import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

from .cones import Cone, ZeroCone, NonnegativeOrthant, SecondOrderCone, PositiveSemidefiniteCone
from .linalg import matvec
from .solver import Solver_Result, Solver_Status


def solve(
    P: Float[Tensor, "n n"],
    q: Float[Tensor, "n"],
    A: Float[Tensor, "m n"],
    b: Float[Tensor, "m"],
    cones: list[Cone],
    max_iter: int = 50,
    time_limit: float = np.inf,
    conv_tol: float = 1e-5,
    verbose: bool = False,
) -> tuple:
    """Solve the cone problem with Clarabel and return `(x_opt, y_opt, obj_val, status)`."""
    x_opt, y_opt, obj_val, status, _stats = solve_with_stats(
        P,
        q,
        A,
        b,
        cones,
        max_iter=max_iter,
        time_limit=time_limit,
        conv_tol=conv_tol,
        verbose=verbose,
    )
    return x_opt, y_opt, obj_val, status


def solve_with_stats(
    P: Float[Tensor, "n n"],
    q: Float[Tensor, "n"],
    A: Float[Tensor, "m n"],
    b: Float[Tensor, "m"],
    cones: list[Cone],
    max_iter: int = 50,
    time_limit: float = np.inf,
    conv_tol: float = 1e-5,
    verbose: bool = False,
) -> tuple:
    """Solve the cone problem with Clarabel and return the solution data and solver statistics.

    Builds a new `DefaultSolver` per call with Clarabel's default settings,
    except for the iteration limit, time limit, and tolerances.
    One-shot counterpart to :class:`ClarabelSCPSolver`.

    Returns `(x_opt, y_opt, obj_val, status, stats)`
    with `status == "optimal"` on success and `x_opt is None` otherwise.
    """
    settings = clarabel_native.DefaultSettings()
    settings.verbose = verbose
    settings.max_iter = max_iter
    settings.time_limit = time_limit
    settings.tol_gap_abs = conv_tol
    settings.tol_gap_rel = conv_tol
    settings.tol_feas = conv_tol

    P_triu = sp.triu(_to_csc(P)).tocsc()
    A_csc = _to_csc(A)
    solver = clarabel_native.DefaultSolver(
        P_triu, q.detach().cpu().numpy(), A_csc, b.detach().cpu().numpy(), _clarabel_cones(cones), settings
    )
    sol = solver.solve()

    stats = {
        "solver_name": "Clarabel (native)",
        "solve_time": float(sol.solve_time),
        "setup_time": 0.0,  # solve_time includes equilibration/factorization
        "num_iters": int(sol.iterations),
        "extra_stats": str(sol.status),
    }
    if sol.status != clarabel_native.SolverStatus.Solved:
        return None, None, None, str(sol.status), stats

    x_opt = torch.as_tensor(np.asarray(sol.x, dtype=np.float64))
    y_opt = torch.as_tensor(np.asarray(sol.z, dtype=np.float64))
    return x_opt, y_opt, float(sol.obj_val), "optimal", stats


def make_solver_result(x_opt, y_opt, A, b, obj_val) -> Solver_Result:
    """Wrap a single optimal point as a 1-iterate `Solver_Result`."""
    n = x_opt.shape[0]
    m = A.shape[0]
    z_opt = b - matvec(A, x_opt)

    return Solver_Result(
        x_hist=x_opt.reshape(1, n),
        z_hist=z_opt.reshape(1, m),
        y_hist=y_opt.reshape(1, m),
        rhos_hist=torch.zeros_like(y_opt.reshape(1, m)),
        alpha_hist=torch.zeros(1),
        obj_val_hist=torch.tensor([obj_val]),
        dual_res_norm_hist=torch.zeros(1),
        prim_res_norm_hist=torch.zeros(1),
        status=Solver_Status.CONVERGED,
        solve_time=0.0,
        setup_time=0.0,
        coeff_fac_iters=[],
        cg_iters=[],
    )


def _to_csc(T: Tensor) -> sp.csc_matrix:
    """Convert a (possibly sparse, possibly GPU) torch matrix to a SciPy CSC matrix.

    Sparse layouts convert index-for-index, without densifying.
    """
    T = T.detach()
    if T.layout == torch.sparse_csr:
        mat = sp.csr_matrix(
            (T.values().cpu().numpy(), T.col_indices().cpu().numpy(), T.crow_indices().cpu().numpy()),
            shape=tuple(T.shape),
        )
    elif T.layout == torch.sparse_csc:
        mat = sp.csc_matrix(
            (T.values().cpu().numpy(), T.row_indices().cpu().numpy(), T.ccol_indices().cpu().numpy()),
            shape=tuple(T.shape),
        )
    elif T.layout == torch.sparse_coo:
        C = T.coalesce()
        idx = C.indices().cpu().numpy()
        mat = sp.coo_matrix((C.values().cpu().numpy(), (idx[0], idx[1])), shape=tuple(C.shape))
    else:
        mat = T.cpu().numpy()
    return sp.csc_matrix(mat)


def _clarabel_cones(cones: list[Cone]) -> list:
    """Translate this package's cones into Clarabel's native cone objects, preserving the order.

    PSD cones are passed as `PSDTriangleConeT(k)` where `k` is the matrix size.
    The `svec` ordering (column-major upper triangle, off-diagonals scaled by `sqrt(2)`)
    matches Clarabel's triangular convention.
    """
    out = []
    for cone in cones:
        size = int(cone.indices.numel())
        if isinstance(cone, ZeroCone):
            out.append(clarabel_native.ZeroConeT(size))
        elif isinstance(cone, NonnegativeOrthant):
            out.append(clarabel_native.NonnegativeConeT(size))
        elif isinstance(cone, SecondOrderCone):
            out.append(clarabel_native.SecondOrderConeT(size))
        elif isinstance(cone, PositiveSemidefiniteCone):
            k = (int(np.sqrt(1 + 8 * size)) - 1) // 2
            out.append(clarabel_native.PSDTriangleConeT(k))
        else:
            raise ValueError(f"Unsupported cone type for Clarabel: {cone.__class__.__name__}")
    return out


class ClarabelSCPSolver:
    """Clarabel solver that reuses its setup across SCP outer iterations.

    Across outer SCP iterations, the subproblems keep a fixed cone structure and sparsity pattern of `(P, A)`,
    with only the numerical values changing as e.g. the dynamics are relinearized.
    Thus, the solver can be constructed once, and every subsequent solve can call `DefaultSolver.update(...)`
    to update the data values in-place, allowing the reuse of the equilibration and symbolic KKT factorization.

    A solve falls back to a full rebuild whenever the sparsity pattern changes
    or when Clarabel reports that an in-place update is not allowed.

    NOTE: in-place data updates require Clarabel's presolve and structural reductions to be disabled.

    `solve_with_stats` mirrors the module-level function of the same name
    so it is a drop-in replacement for SCP experiments.
    """

    def __init__(self, max_iter: int = 200, time_limit: float = np.inf, conv_tol: float = 1e-5, verbose: bool = False):
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.conv_tol = conv_tol
        self.verbose = verbose
        self._solver: clarabel_native.DefaultSolver | None = None
        self._cones: list | None = None
        self._P_pattern: tuple[np.ndarray, np.ndarray] | None = None  # (indices, indptr) of triu(P)
        self._A_pattern: tuple[np.ndarray, np.ndarray] | None = None  # (indices, indptr) of A

    def _settings(self) -> clarabel_native.DefaultSettings:
        s = clarabel_native.DefaultSettings()
        s.verbose = self.verbose
        s.max_iter = self.max_iter
        s.time_limit = self.time_limit
        s.tol_gap_abs = self.conv_tol
        s.tol_gap_rel = self.conv_tol
        s.tol_feas = self.conv_tol
        # Required for in-place data updates as these reductions change the problem structure
        s.presolve_enable = False
        s.chordal_decomposition_enable = False
        return s

    @staticmethod
    def _pattern_eq(a: tuple[np.ndarray, np.ndarray] | None, b: tuple[np.ndarray, np.ndarray]) -> bool:
        return a is not None and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def solve_with_stats(
        self,
        P: Float[Tensor, "n n"],
        q: Float[Tensor, "n"],
        A: Float[Tensor, "m n"],
        b: Float[Tensor, "m"],
        cones: list[Cone],
    ):
        """Solve `(P, q, A, b, cones)`, reusing the prior setup when the pattern is unchanged.

        Returns `(x_opt, y_opt, obj_val, status, stats)` matching the module-level `solve_with_stats` interface.
        """
        P_triu = sp.triu(_to_csc(P)).tocsc()
        A_csc = _to_csc(A)
        q_np = q.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        P_pat, A_pat = (P_triu.indices, P_triu.indptr), (A_csc.indices, A_csc.indptr)

        can_update = (
            self._solver is not None
            and self._pattern_eq(self._P_pattern, P_pat)
            and self._pattern_eq(self._A_pattern, A_pat)
            and self._solver.is_data_update_allowed()
        )
        if can_update:
            self._solver.update(P=P_triu.data, q=q_np, A=A_csc.data, b=b_np)
        else:
            self._cones = _clarabel_cones(cones)
            self._solver = clarabel_native.DefaultSolver(P_triu, q_np, A_csc, b_np, self._cones, self._settings())
            self._P_pattern, self._A_pattern = P_pat, A_pat

        sol = self._solver.solve()
        stats = {
            "solver_name": "Clarabel (native, reused)",
            "solve_time": float(sol.solve_time),
            "setup_time": 0.0,  # native solve_time already includes equilibration/factorization
            "num_iters": int(sol.iterations),
            "extra_stats": str(sol.status),
        }
        good_statuses = {
            clarabel_native.SolverStatus.Solved,
            clarabel_native.SolverStatus.AlmostSolved,
            clarabel_native.SolverStatus.MaxTime,
        }
        if sol.status not in good_statuses:
            # After a failed solve, the in-place update is not valid
            # The next call will rebuild the solver from scratch
            self._solver = None
            self._P_pattern = None
            self._A_pattern = None
            return None, None, None, str(sol.status), stats

        x_opt = torch.as_tensor(np.asarray(sol.x, dtype=np.float64))
        y_opt = torch.as_tensor(np.asarray(sol.z, dtype=np.float64))
        return x_opt, y_opt, float(sol.obj_val), str(sol.status), stats
