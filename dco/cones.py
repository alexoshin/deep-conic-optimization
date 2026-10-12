"""Convex cones and projections."""

from abc import abstractmethod
from collections.abc import Iterable

from jaxtyping import Float
import torch
from torch import Tensor

from .linalg import min_eigval_gap, smat, svec


def convert_socc_to_conic(
    A: Float[Tensor, "... m n"],
    b: Float[Tensor, "... m"],
    c: Float[Tensor, "... n"],
    d: Float[Tensor, "..."],
) -> tuple[Float[Tensor, "... m+1 n"], Float[Tensor, "... m+1"]]:
    r"""Convert `||A x + b||_2 <= c^T x + d` to the conic inclusion form `b - A^T x \in SOC`."""
    A_conic = torch.cat([-c.unsqueeze(-2), -A], dim=-2)
    b_conic = torch.cat([d.unsqueeze(-1), b], dim=-1)
    return A_conic, b_conic


class Cone(torch.nn.Module):
    """A convex cone over a subset `indices` of the constraint rows."""

    def __init__(self, indices: Iterable[int]) -> None:
        super().__init__()
        self.indices = indices.clone() if isinstance(indices, torch.Tensor) else torch.tensor(indices)
        self.size = self.indices.numel()

    @staticmethod
    @abstractmethod
    def project(z: Float[Tensor, "... k"]) -> Float[Tensor, "... k"]:
        """Euclidean projection of a vector onto the cone."""

    def forward(self, z: Float[Tensor, "... m"]) -> Float[Tensor, "... m"]:
        """Project the rows `indices` of `z` onto the cone, leaving the other rows unchanged."""
        z_proj = z.clone()
        z_proj[..., self.indices] = self.project(z[..., self.indices])
        return z_proj

    def to(self, *args, **kwargs):
        self.indices = self.indices.to(*args, **kwargs)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(indices={self.indices.tolist()}, size={self.size})"


class ZeroCone(Cone):
    """The zero cone `{0}` (equality constraints)."""

    @staticmethod
    def project(z: Float[Tensor, "... k"]) -> Float[Tensor, "... k"]:
        return torch.zeros_like(z)

    def forward(self, z: Float[Tensor, "... m"]) -> Float[Tensor, "... m"]:
        z_proj = z.clone()
        z_proj.index_fill_(-1, self.indices, 0.0)
        return z_proj


class NonnegativeOrthant(Cone):
    """The nonnegative orthant `x >= 0` (inequality constraints)."""

    @staticmethod
    def project(z: Float[Tensor, "... k"]) -> Float[Tensor, "... k"]:
        return torch.clamp(z, min=0.0)


class SecondOrderCone(Cone):
    """The second-order cone `||x||_2 <= t` on the vector `(t, x)`."""

    @staticmethod
    def project(tx: Float[Tensor, "... k+1"]) -> Float[Tensor, "... k+1"]:
        t = tx[..., 0:1]
        x = tx[..., 1:]
        x_norm = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
        tx_proj = torch.zeros_like(tx)  # ||x|| <= -t: the origin
        tx_proj = torch.where(x_norm <= t, tx, tx_proj)  # inside the cone
        scale = (x_norm + t) / 2.0
        denom = torch.clamp(x_norm, min=torch.finfo(x.dtype).tiny)
        return torch.where(x_norm > torch.abs(t), torch.cat([scale, scale * (x / denom)], dim=-1), tx_proj)


class PositiveSemidefiniteCone(Cone):
    """The PSD cone on a symmetric matrix, stored as `svec(X)`.

    `gradient_method` selects the backward pass of the projection:
      - dk (default): the Daleckii-Krein rule of :class:`PSDProject`, which stays finite at repeated eigenvalues
      - autograd: differentiates through `torch.linalg.eigh`, which fails when eigenvalues are repeated
    """

    def __init__(self, indices: Iterable[int], gradient_method: str = "dk") -> None:
        super().__init__(indices)
        if gradient_method not in ("autograd", "dk"):
            raise ValueError(f"gradient_method must be 'autograd' or 'dk', got {gradient_method!r}")
        self.gradient_method = gradient_method

    @staticmethod
    def project(x: Float[Tensor, "... n*(n+1)//2"]) -> Float[Tensor, "... n*(n+1)//2"]:
        """Projection of `smat(x)` onto the PSD cone through `eigh` (plain autograd backward)."""
        X = smat(x)
        X_sym = 0.5 * (X + X.mT)
        eigvals, eigvecs = torch.linalg.eigh(X_sym)
        X_proj = eigvecs @ torch.diag_embed(torch.clamp(eigvals, min=0.0)) @ eigvecs.mT
        return svec(X_proj)

    def project_block(self, z_sub: Float[Tensor, "... k"]) -> Float[Tensor, "... k"]:
        """Projection of one (or a batch of) `svec` blocks using `gradient_method`."""
        if self.gradient_method == "autograd":
            return self.project(z_sub)
        return svec(PSDProject.apply(smat(z_sub)))

    def forward(self, z: Float[Tensor, "... m"]) -> Float[Tensor, "... m"]:
        z_proj = z.clone()
        z_proj[..., self.indices] = self.project_block(z[..., self.indices])
        return z_proj

    def min_eigval_gaps(self, z: Float[Tensor, "... m"]) -> Float[Tensor, "..."]:
        """Minimum eigenvalue gap of the matrix this cone would project (for diagnostics)."""
        return min_eigval_gap(smat(z[..., self.indices]))


class PSDProject(torch.autograd.Function):
    """Matrix-form projection onto the PSD cone with Daleckii-Krein backward rule.

    Forward: `Pi(S) = U max(Lambda, 0) U^T` for symmetric `S = U Lambda U^T`.

    Backward: the projection is a primary matrix function of the form `f(x) = max(x, 0)`,
    whose Frechet derivative is given by the Daleckii-Krein theorem.
    For a symmetric cotangent `G` of the output, the cotangent of the input is `U (L * (U^T G U)) U^T`
    with the Loewner matrix of first divided differences `L_ij = (f(l_j) - f(l_i)) / (l_j - l_i)`
    for `l_i != l_j` and `L_ii = f'(l_i)`. The divided difference is bounded by 1,
    so the rule is finite at coalescing eigenvalues, where autograd through `eigh` diverges.
    The generalized derivative at `lambda = 0` is taken as `f'(0) = 0`, matching `torch.relu`.
    """

    @staticmethod
    def forward(ctx, A):  # type: ignore[override]
        A_sym = 0.5 * (A + A.mT)
        eigvals, eigvecs = torch.linalg.eigh(A_sym)
        Y = eigvecs @ torch.diag_embed(torch.clamp(eigvals, min=0.0)) @ eigvecs.mT
        ctx.save_for_backward(eigvals, eigvecs)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):  # type: ignore[override]
        eigvals, eigvecs = ctx.saved_tensors
        G = 0.5 * (grad_Y + grad_Y.mT)
        f = torch.clamp(eigvals, min=0.0)
        diff = eigvals.unsqueeze(-1) - eigvals.unsqueeze(-2)
        same = diff == 0.0
        safe_diff = torch.where(same, torch.ones_like(diff), diff)
        L_offdiag = (f.unsqueeze(-1) - f.unsqueeze(-2)) / safe_diff
        deriv = (eigvals > 0).to(L_offdiag.dtype)
        L_diag = 0.5 * (deriv.unsqueeze(-1) + deriv.unsqueeze(-2))
        L = torch.where(same, L_diag, L_offdiag)
        grad_A = eigvecs @ (L * (eigvecs.mT @ G @ eigvecs)) @ eigvecs.mT
        return 0.5 * (grad_A + grad_A.mT)


def psd_project(A: Float[Tensor, "... n n"]) -> Float[Tensor, "... n n"]:
    """Projection of a symmetric matrix onto the PSD cone using the Daleckii-Krein backward rule."""
    return PSDProject.apply(A)


# ----- Batched cones and projections -----
# Multiple same-type cones of a super-batch projected by a single batched operation,
# so the per-cone Python and kernel-launch overhead does not grow with the batch size


class BatchedZeroCone(ZeroCone):
    def __init__(self, indices: Tensor) -> None:
        super().__init__(indices.flatten())


class BatchedNonnegativeOrthant(NonnegativeOrthant):
    def __init__(self, indices: Tensor) -> None:
        super().__init__(indices.flatten())


class BatchedSecondOrderCone(SecondOrderCone):
    """Many SOCs of one dimension, projected as a `(B_cone, k+1)` tensor."""

    def __init__(self, indices_2d: Tensor) -> None:
        super().__init__(indices_2d.flatten())
        self.indices_2d = indices_2d

    def forward(self, z: Float[Tensor, "... m"]) -> Float[Tensor, "... m"]:
        z_proj = z.clone()
        z_proj[..., self.indices_2d] = SecondOrderCone.project(z[..., self.indices_2d])
        return z_proj

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.indices_2d = self.indices_2d.to(*args, **kwargs)
        return self


class PaddedBatchedSecondOrderCone(SecondOrderCone):
    """SOCs of different dimensions projected as one zero-padded `(B_cone, k_max)` tensor.

    Padded slots use `(t, x) = (0, 0)`, a fixed point of the projection,
    and only the valid prefix of each row is written back.
    """

    def __init__(self, indices_list: list[Tensor]) -> None:
        if not indices_list:
            raise ValueError("PaddedBatchedSecondOrderCone needs at least one cone")
        k_max = max(idx.numel() for idx in indices_list)
        device = indices_list[0].device
        padded = torch.zeros(len(indices_list), k_max, dtype=torch.long, device=device)
        mask = torch.zeros(len(indices_list), k_max, dtype=torch.bool, device=device)
        for i, idx in enumerate(indices_list):
            padded[i, : idx.numel()] = idx
            mask[i, : idx.numel()] = True
        super().__init__(torch.cat([idx.flatten() for idx in indices_list]))
        self.padded_indices = padded
        self.mask = mask
        self.valid_row, self.valid_col = torch.where(self.mask)

    def forward(self, z: Float[Tensor, "... m"]) -> Float[Tensor, "... m"]:
        z_proj = z.clone()
        proj = SecondOrderCone.project(z[..., self.padded_indices] * self.mask) * self.mask
        z_proj[..., self.indices] = proj[..., self.valid_row, self.valid_col]
        return z_proj

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for name in ("padded_indices", "mask", "valid_row", "valid_col"):
            setattr(self, name, getattr(self, name).to(*args, **kwargs))
        return self


class BatchedPositiveSemidefiniteCone(PositiveSemidefiniteCone):
    """Many PSD cones of one matrix size, projected through one batched `eigh`."""

    def __init__(self, indices_2d: Tensor, gradient_method: str = "dk") -> None:
        super().__init__(indices_2d.flatten(), gradient_method=gradient_method)
        self.indices_2d = indices_2d

    def forward(self, z: Float[Tensor, "... m"]) -> Float[Tensor, "... m"]:
        z_proj = z.clone()
        z_proj[..., self.indices_2d] = self.project_block(z[..., self.indices_2d])
        return z_proj

    def min_eigval_gaps(self, z: Float[Tensor, "... m"]) -> Float[Tensor, "... B_cone"]:
        return min_eigval_gap(smat(z[..., self.indices_2d]))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.indices_2d = self.indices_2d.to(*args, **kwargs)
        return self


BATCHED_CONE_TYPES = (
    BatchedZeroCone,
    BatchedNonnegativeOrthant,
    BatchedSecondOrderCone,
    PaddedBatchedSecondOrderCone,
    BatchedPositiveSemidefiniteCone,
)

SOC_STRATEGIES = ("auto", "same_dim", "padded")


def resolve_soc_strategy(soc_strategy: str, soc_sizes: Iterable[int]) -> str:
    """`auto` pads when SOCs have two or more distinct sizes (results in a single pass rather than one per size)."""
    if soc_strategy not in SOC_STRATEGIES:
        raise ValueError(f"Unknown soc_strategy: {soc_strategy}")
    if soc_strategy != "auto":
        return soc_strategy
    return "padded" if len(set(soc_sizes)) >= 2 else "same_dim"


def build_super_cones(
    cones_per_problem: list[list[Cone]],
    m_offsets: list[int],
    soc_strategy: str = "auto",
) -> list[Cone]:
    """Build each problem's cones as a single flat super-vector and group them by type.

    Problems can differ in size and cone structure.
    Each one carries its own cone list, shifted by where its constraint slice starts.
    The result projects every cone type with one batched op.
    """
    if len(cones_per_problem) != len(m_offsets):
        raise ValueError("cones_per_problem and m_offsets must have the same length")
    first = next((c for c in cones_per_problem if c), None)
    device = first[0].indices.device if first else torch.get_default_device()

    zero_idxs: list[Tensor] = []
    nonneg_idxs: list[Tensor] = []
    soc_by_size: dict[int, list[Tensor]] = {}
    soc_all: list[Tensor] = []
    psd_by_size: dict[int, list[Tensor]] = {}
    psd_gradient_method: dict[int, str] = {}
    for cones_b, offset in zip(cones_per_problem, m_offsets):
        for cone in cones_b:
            idx = cone.indices.to(device) + offset
            if isinstance(cone, ZeroCone):
                zero_idxs.append(idx)
            elif isinstance(cone, NonnegativeOrthant):
                nonneg_idxs.append(idx)
            elif isinstance(cone, SecondOrderCone):
                soc_by_size.setdefault(cone.size, []).append(idx)
                soc_all.append(idx)
            elif isinstance(cone, PositiveSemidefiniteCone):
                psd_by_size.setdefault(cone.size, []).append(idx)
                psd_gradient_method[cone.size] = cone.gradient_method
            else:
                raise ValueError(f"Unknown cone type: {cone.__class__.__name__}")

    super_cones: list[Cone] = []
    if zero_idxs:
        super_cones.append(BatchedZeroCone(torch.cat(zero_idxs)))
    if nonneg_idxs:
        super_cones.append(BatchedNonnegativeOrthant(torch.cat(nonneg_idxs)))
    if resolve_soc_strategy(soc_strategy, soc_by_size.keys()) == "same_dim":
        for size in sorted(soc_by_size):
            super_cones.append(BatchedSecondOrderCone(torch.stack(soc_by_size[size])))
    elif soc_all:
        super_cones.append(PaddedBatchedSecondOrderCone(soc_all))
    for size in sorted(psd_by_size):
        super_cones.append(BatchedPositiveSemidefiniteCone(torch.stack(psd_by_size[size]), psd_gradient_method[size]))
    return super_cones
