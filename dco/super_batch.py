"""Super-batched problem helpers."""

import torch
from torch import Tensor


def flat_super_matvec(M_sparse: Tensor, v: Tensor) -> Tensor:
    """Matrix-vector product between a block-diagonal sparse super-matrix and a flat super-vector."""
    in_total = v.shape[-1]
    lead = v.shape[:-1]
    v_cols = v.reshape(-1, in_total).T
    Mv = M_sparse @ v_cols
    return Mv.T.reshape(*lead, M_sparse.shape[0])


class SegmentReducer:
    """Per-problem reductions over a flat super-vector partitioned into `B` segments.

    `sizes[b]` is the length of sub-problem `b`'s contiguous slice.
    The segments are in order, so the flat vector has length `sum(sizes)`.
    """

    def __init__(self, sizes: list[int], device: torch.device) -> None:
        self.sizes = tuple(int(s) for s in sizes)
        self.B = len(self.sizes)
        self.total = sum(self.sizes)
        self.uniform = len(set(self.sizes)) <= 1
        self.per = self.sizes[0] if self.uniform and self.sizes else 0
        sizes_t = torch.tensor(self.sizes, device=device, dtype=torch.long)
        # Segment id of every flat entry: [0]*sizes[0] + [1]*sizes[1] + ... (length `total`)
        self.seg = torch.repeat_interleave(torch.arange(self.B, device=device), sizes_t)

    def grid(self, v: Tensor) -> Tensor:
        """View a flat super-vector as a `(..., B, per)` grid (only supports uniform)."""
        return v.reshape(*v.shape[:-1], self.B, self.per)

    def sum(self, v: Tensor) -> Tensor:
        """Segment sum over the last dimension: `(..., total) -> (..., B)`."""
        if self.uniform:
            return self.grid(v).sum(dim=-1)
        out = v.new_zeros(*v.shape[:-1], self.B)
        return out.scatter_add(-1, self.seg.expand(v.shape), v)

    def dot(self, a: Tensor, b: Tensor) -> Tensor:
        """Per-segment dot product `a_b . b_b`: `(..., total) -> (..., B)`."""
        if self.uniform:
            return torch.einsum("...i,...i->...", self.grid(a), self.grid(b))
        return self.sum(a * b)

    def inf_norm(self, v: Tensor) -> Tensor:
        """Per-segment infinity norm: `(..., total) -> (..., B)`."""
        if self.uniform:
            if self.per == 0:
                return v.new_zeros(*v.shape[:-1], self.B)
            return self.grid(v).abs().max(dim=-1).values
        out = v.new_zeros(*v.shape[:-1], self.B)
        return out.scatter_reduce(-1, self.seg.expand(v.shape), v.abs(), reduce="amax", include_self=True)

    def max_inf_norm(self, *vs: Tensor) -> Tensor:
        """Per-segment infinity norm over several flat super-vectors at once: `-> (..., B)`."""
        if self.uniform and self.per > 0:
            return torch.cat([self.grid(v) for v in vs], dim=-1).abs().max(dim=-1).values
        reduced = self.inf_norm(vs[0])
        for v in vs[1:]:
            reduced = torch.maximum(reduced, self.inf_norm(v))
        return reduced

    def broadcast(self, per_problem: Tensor) -> Tensor:
        """Inverse of a reduction: `(..., B) -> (..., total)`.

        Used for elementwise multiplication of a per-problem `(B,)` quantity with a flat super-vector.
        """
        if self.uniform:
            return per_problem.unsqueeze(-1).expand(*per_problem.shape, self.per).reshape(*per_problem.shape[:-1], -1)
        return per_problem.index_select(-1, self.seg)


class SuperBatchPadder:
    """Maps a flat super-vector to/from a zero-padded `(B, max_size)` grid."""

    def __init__(self, sizes: list[int], device: torch.device) -> None:
        self.sizes = tuple(int(s) for s in sizes)
        self.B = len(self.sizes)
        self.total = sum(self.sizes)
        self.max_size = max(self.sizes) if self.sizes else 0
        self.sizes_t = torch.tensor(self.sizes, device=device, dtype=torch.long)
        chunks = [torch.arange(s, device=device) + b * self.max_size for b, s in enumerate(self.sizes)]
        self.flat_to_pad = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long, device=device)

    def to_pad(self, v: Tensor) -> Tensor:
        """`(..., total) -> (..., B, max_size)`, zero-padded."""
        lead = v.shape[:-1]
        flat = v.new_zeros(*lead, self.B * self.max_size)
        flat = flat.index_copy(-1, self.flat_to_pad, v)
        return flat.reshape(*lead, self.B, self.max_size)

    def to_flat(self, vp: Tensor) -> Tensor:
        """`(..., B, max_size) -> (..., total)` (drops the padding)."""
        lead = vp.shape[:-2]
        flat = vp.reshape(*lead, self.B * self.max_size)
        return flat.index_select(-1, self.flat_to_pad)


class BatchMetadata:
    """Metadata for a super-batched problem.

    A super-batched problem stacks `B` independent sub-problems block-diagonally into one sparse instance.
    The variables are represented as flat super-vectors: `x` has length `total_n = sum(n_pers)`,
    where `n_pers[b]` is sub-problem `b`'s variable count, and similarly for the constraints.
    """

    def __init__(
        self,
        B: int,
        n_per: int | None = None,
        m_per: int | None = None,
        *,
        n_pers: "list[int] | tuple[int, ...] | None" = None,
        m_pers: "list[int] | tuple[int, ...] | None" = None,
        soc_strategy: str = "auto",
    ) -> None:
        self.B = B
        if n_pers is None:
            if n_per is None:
                raise ValueError("BatchMetadata requires either `n_per` or `n_pers`")
            n_pers = (int(n_per),) * B
        if m_pers is None:
            if m_per is None:
                raise ValueError("BatchMetadata requires either `m_per` or `m_pers`")
            m_pers = (int(m_per),) * B
        self.n_pers = tuple(int(x) for x in n_pers)
        self.m_pers = tuple(int(x) for x in m_pers)
        if len(self.n_pers) != B or len(self.m_pers) != B:
            raise ValueError(f"n_pers/m_pers must have length B={B}, got {len(self.n_pers)}/{len(self.m_pers)}")
        self.soc_strategy = soc_strategy
        self._reducers: dict = {}
        self._padders: dict = {}

    @property
    def uniform(self) -> bool:
        """True when every sub-problem shares the same `(n, m)` (the regular-grid case)."""
        return len(set(self.n_pers)) <= 1 and len(set(self.m_pers)) <= 1

    @property
    def n_per(self) -> int:
        if len(set(self.n_pers)) != 1:
            raise AttributeError("n_per is undefined for a ragged super-batch; use n_pers")
        return self.n_pers[0]

    @property
    def m_per(self) -> int:
        if len(set(self.m_pers)) != 1:
            raise AttributeError("m_per is undefined for a ragged super-batch; use m_pers")
        return self.m_pers[0]

    @property
    def total_n(self) -> int:
        return sum(self.n_pers)

    @property
    def total_m(self) -> int:
        return sum(self.m_pers)

    @property
    def m_offsets(self) -> list[int]:
        """The start index of each sub-problem's constraint slice in the flat super-vector."""
        offsets, acc = [], 0
        for m in self.m_pers:
            offsets.append(acc)
            acc += m
        return offsets

    def reducers(self, device: "torch.device"):
        """Return `(reducer_n, reducer_m)` :class:`SegmentReducer`s for this device (cached)."""
        key = str(device)
        if key not in self._reducers:
            self._reducers[key] = (
                SegmentReducer(self.n_pers, device),
                SegmentReducer(self.m_pers, device),
            )
        return self._reducers[key]

    def padders(self, device: "torch.device"):
        """Return `(padder_n, padder_m)` :class:`SuperBatchPadder`s for this device (cached)."""
        key = str(device)
        if key not in self._padders:
            self._padders[key] = (
                SuperBatchPadder(self.n_pers, device),
                SuperBatchPadder(self.m_pers, device),
            )
        return self._padders[key]
