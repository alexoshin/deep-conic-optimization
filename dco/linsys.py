"""Reduced linear system solve and matrix-free backward rule."""

from jaxtyping import Float
import torch
from torch import Tensor

from .cg import LinearOperator, Preconditioner, conjugate_gradient, identity_preconditioner
from .linalg import add_diag, dot_prod, matvec
from .structure import MatrixStructure, detect_matrix_structure


class AdjointWarmStartCache:
    """Cache of the previous adjoint CG solution.

    For an unrolled trajectory of K layers, the adjoint CG solve is called backwards-in-time with a slowly varying operator,
    so the previous adjoint solution can be used as a good initial guess for the current one.
    """

    __slots__ = ("lam", "hits", "misses")

    def __init__(self) -> None:
        self.lam: Tensor | None = None
        self.hits = 0
        self.misses = 0

    def take(self, rhs: Float[Tensor, "... n"]) -> Float[Tensor, "... n"] | None:
        lam = self.lam
        if lam is not None and lam.shape == rhs.shape and lam.dtype == rhs.dtype and lam.device == rhs.device:
            self.hits += 1
            return lam
        self.misses += 1
        return None

    def store(self, lam: Float[Tensor, "... n"]) -> None:
        self.lam = lam.detach()


class ReducedSystemOperator(LinearOperator):
    """Matrix-free operator `M(rho) = P + sigma I + A^T diag(rho) A`.

    The sparsity of `P` is analyzed via :func:`dco.structure.detect_matrix_structure`.
    `A_T` and `A_sq_T` accept the transposes that :class:`dco.problem.SuperProblem` has already built.
    """

    def __init__(
        self,
        P: Float[Tensor, "... n n"],
        A: Float[Tensor, "... m n"],
        rhos: Float[Tensor, "... m"],
        sigma: float = 0.0,
        structure_dispatch: bool = True,
        A_T: Float[Tensor, "... n m"] | None = None,
        A_sq_T: Float[Tensor, "... n m"] | None = None,
    ) -> None:
        self.sigma = sigma
        self.structure = detect_matrix_structure(P) if structure_dispatch else MatrixStructure.SPARSE
        self.A = A
        if A_T is None:
            A_T = A.mT if A.layout == torch.strided else A.mT.to_sparse_csr()
        self.A_T = A_T
        self.rhos = rhos
        if self.structure is MatrixStructure.SPARSE:
            if sigma == 0.0:
                self.P_sigmaI = P
            else:
                self.P_sigmaI = add_diag(P, sigma * torch.ones(P.shape[-1], dtype=P.dtype, device=P.device))
            self.diag_P_sigmaI = self.extract_diagonal(self.P_sigmaI)
        else:
            self.P_sigmaI = None
            if self.structure is MatrixStructure.ZERO:
                self.diag_P_sigmaI = sigma * torch.ones(P.shape[-1], dtype=P.dtype, device=P.device)
            else:
                self.diag_P_sigmaI = self.extract_diagonal(P) + sigma
        if A_sq_T is not None:
            self.A_sq_T = A_sq_T
        elif A.layout == torch.strided:
            self.A_sq_T = A.pow(2).mT
        else:
            A_csr = A.to_sparse_csr()
            A_sq = torch.sparse_csr_tensor(
                A_csr.crow_indices(), A_csr.col_indices(), A_csr.values().pow(2), size=A_csr.shape
            )
            self.A_sq_T = A_sq.mT.to_sparse_csr()

    def matvec(self, x: Float[Tensor, "... n"]) -> Float[Tensor, "... n"]:
        Atop_rhoAx = matvec(self.A_T, self.rhos * matvec(self.A, x))
        if self.structure is MatrixStructure.SPARSE:
            return matvec(self.P_sigmaI, x) + Atop_rhoAx
        return self.diag_P_sigmaI * x + Atop_rhoAx

    def update_rhos(self, rhos: Float[Tensor, "... m"]) -> None:
        self.rhos = rhos

    def with_rhos(self, rhos: Float[Tensor, "... m"]) -> "ReducedSystemOperator":
        """Create a copy of this operator with a new `rhos` vector, while still sharing the rest of the data.

        This is necessary because each backward solve must use the same `rho` as the forward solve.
        """
        clone = object.__new__(type(self))
        clone.sigma = self.sigma
        clone.structure = self.structure
        clone.P_sigmaI = self.P_sigmaI
        clone.A = self.A
        clone.A_T = self.A_T
        clone.rhos = rhos
        clone.diag_P_sigmaI = self.diag_P_sigmaI
        clone.A_sq_T = self.A_sq_T
        return clone

    def diagonal(self, rhos: Float[Tensor, "... m"] | None = None) -> Float[Tensor, "... n"]:
        """Compute the diagonal `diag(M) = diag(P + sigma I) + sum_j rho_j A_ji^2`."""
        return self.diag_P_sigmaI + matvec(self.A_sq_T, self.rhos if rhos is None else rhos)

    @staticmethod
    def extract_diagonal(M: Float[Tensor, "... n n"]) -> Float[Tensor, "... n"]:
        if M.layout == torch.strided:
            return torch.diagonal(M, dim1=-2, dim2=-1)
        M_coo = M.to_sparse_coo().coalesce()
        idx, vals = M_coo.indices(), M_coo.values()
        diag = torch.zeros(M.shape[-1], dtype=M.dtype, device=M.device)
        on_diag = idx[0] == idx[1]
        diag[idx[0][on_diag]] = vals[on_diag]
        return diag


def solve_reduced_system(
    linop: ReducedSystemOperator,
    rhos: Float[Tensor, "... m"],
    rhs: Float[Tensor, "... n"],
    x0: Float[Tensor, "... n"] | None = None,
    preconditioner: Preconditioner = identity_preconditioner,
    max_iter: int | None = None,
    eps_abs: float = 1e-6,
    eps_rel: float = 0.0,
    backward_eps_abs: float | None = None,
    backward_eps_rel: float | None = None,
    backward_max_iter: int | None = None,
    adjoint_cache: AdjointWarmStartCache | None = None,
    check_every: int = 1,
) -> tuple[Float[Tensor, "... n"], int]:
    """Solve `M(rho) x = rhs` by CG with the matrix-free implicit layer :class:`ImplicitCGLayer`.

    The `backward_*` settings apply to the adjoint solve (`None` inherits the forward setting).
    Passing `adjoint_cache` warm-starts the solve from the previous layer's adjoint solution.
    Returns `(x, cg_iterations)`.
    """
    return ImplicitCGLayer.apply(
        linop,
        rhos,
        rhs,
        x0,
        preconditioner,
        max_iter,
        eps_abs,
        eps_rel,
        backward_eps_abs,
        backward_eps_rel,
        backward_max_iter,
        adjoint_cache,
        check_every,
    )  # type: ignore


class ImplicitCGLayer(torch.autograd.Function):
    """Matrix-free CG solve of `M(rho) x = rhs`."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        linop: ReducedSystemOperator,
        rhos: Float[Tensor, "... m"],
        rhs: Float[Tensor, "... n"],
        x0: Float[Tensor, "... n"] | None,
        preconditioner: Preconditioner,
        max_iter: int | None,
        eps_abs: float,
        eps_rel: float,
        backward_eps_abs: float | None,
        backward_eps_rel: float | None,
        backward_max_iter: int | None,
        adjoint_cache: AdjointWarmStartCache | None,
        check_every: int,
    ) -> tuple[Float[Tensor, "... n"], int]:
        """Forward pass of the implicit CG layer, solving `M(rho) x = rhs`."""

        # Cache a copy of the current rho, since the backward pass must use the same rho as the forward pass.
        # rhos is detached because the gradient with respect to rho will be returned explicitly,
        # and an attached rho forms a cycle through ctx that keeps the graph alive and prevents memory from being freed.
        layer_linop = linop.with_rhos(rhos.detach())

        # Solve the linear system M(rho) x = rhs using CG
        x, cg_iters = conjugate_gradient(
            layer_linop,
            rhs,
            x0=x0,
            preconditioner=preconditioner,
            max_iter=max_iter,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            check_every=check_every,
        )

        ctx.save_for_backward(x0, x)
        ctx.linop = layer_linop
        ctx.preconditioner = preconditioner
        ctx.max_iter = backward_max_iter if backward_max_iter is not None else max_iter
        ctx.eps_abs = backward_eps_abs if backward_eps_abs is not None else eps_abs
        ctx.eps_rel = backward_eps_rel if backward_eps_rel is not None else eps_rel
        ctx.adjoint_cache = adjoint_cache
        ctx.check_every = check_every

        return x, cg_iters

    @staticmethod
    def backward(ctx, L_x: Float[Tensor, "... n"], L_iters: None):  # type: ignore[override]
        """Backward pass of the implicit CG layer, computing the gradient of the loss with respect to rho and rhs."""

        x0, x = ctx.saved_tensors
        linop, preconditioner, cache = ctx.linop, ctx.preconditioner, ctx.adjoint_cache

        # Solve the adjoint system M lambda = -dL/dx
        rhs = -L_x
        adjoint_x0 = cache.take(rhs) if cache is not None else None
        tol_ref_sq = dot_prod(rhs, preconditioner(rhs)) if adjoint_x0 is not None else None
        lam, _ = conjugate_gradient(
            linop,
            rhs,
            x0=adjoint_x0,
            preconditioner=preconditioner,
            max_iter=ctx.max_iter,
            eps_abs=ctx.eps_abs,
            eps_rel=ctx.eps_rel,
            tol_ref_sq=tol_ref_sq,
            check_every=ctx.check_every,
        )

        if cache is not None:
            cache.store(lam)

        # dM / drho_i = a_i a_i^T, so dL/drho_i = lambda^T a_i a_i^T x = (A lambda)_i (A x)_i.
        L_rhos = matvec(linop.A, lam) * matvec(linop.A, x)
        L_rhs = -lam
        L_x0 = torch.zeros_like(x0) if x0 is not None else None

        return None, L_rhos, L_rhs, L_x0, None, None, None, None, None, None, None, None, None
