"""Datasets of cone programs with reference solutions.

Each instance is a pickle `problem_*.pkl` holding `(P, q, A, b, cones, result)`,
where `result` is a :class:`dco.solver.Solver_Result` with the reference solution as its last iterate.
"""

from collections.abc import Iterable
import pathlib
import pickle

import torch
from torch import Tensor

from .cones import Cone, PositiveSemidefiniteCone
from .sparse import block_diag_csr
from .super_batch import BatchMetadata

# (P, q, A, b, cones, x_opt, s_opt, y_opt) of one problem or a dense batch of same-shaped problems.
Sample = tuple[Tensor, Tensor, Tensor, Tensor, list[Cone], Tensor, Tensor, Tensor]

# Super-batched counterpart: block-diagonal sparse P, A; flat q, b and optima; one cone list per sub-problem.
SuperSample = tuple[Tensor, Tensor, Tensor, Tensor, list[list[Cone]], Tensor, Tensor, Tensor]


def problem_files(directory: pathlib.Path) -> list[pathlib.Path]:
    """The instance files of a dataset directory, in sorted order."""
    return sorted(directory.glob("problem_*.pkl"))


class ConeProblemDataset:
    """A list of instance files, loaded on access onto the default device and dtype."""

    def __init__(self, problem_paths: list[pathlib.Path], gradient_method: str = "dk") -> None:
        """`gradient_method` is applied to every PSD cone loaded, overriding what the pickle stored."""
        if gradient_method not in ("autograd", "dk"):
            raise ValueError("gradient_method must be 'autograd' or 'dk'")
        self.problem_paths = list(problem_paths)
        self.gradient_method = gradient_method

    def __len__(self) -> int:
        return len(self.problem_paths)

    def __getitem__(self, idx: int) -> Sample:
        kwargs = {"device": torch.get_default_device(), "dtype": torch.get_default_dtype()}
        with self.problem_paths[idx].open("rb") as f:
            P, q, A, b, cones, result = pickle.load(f)
        for cone in cones:
            if isinstance(cone, PositiveSemidefiniteCone):
                cone.gradient_method = self.gradient_method
        return (
            P.to(**kwargs),
            q.to(**kwargs),
            A.to(**kwargs),
            b.to(**kwargs),
            cones,
            result.x.to(**kwargs),
            result.z.to(**kwargs),
            result.y.to(**kwargs),
        )

    def sample_batch(self, indices: Iterable[int]) -> Sample:
        """A dense `(B, ...)` batch; the problems must share their shape and cone structure."""
        items = [self[int(i)] for i in indices]
        P = torch.stack([it[0].to_dense() for it in items])
        A = torch.stack([it[2].to_dense() for it in items])
        stack = lambda j: torch.stack([it[j] for it in items])  # noqa: E731
        return P, stack(1), A, stack(3), items[0][4], stack(5), stack(6), stack(7)

    def yield_batches(self, batch_size: int, shuffle: bool = True) -> Iterable[Sample]:
        indices = torch.randperm(len(self)) if shuffle else torch.arange(len(self))
        for i in range(0, len(self), batch_size):
            yield self.sample_batch(indices[i : i + batch_size].tolist())

    def sample_super_batch(
        self, indices: Iterable[int], soc_strategy: str = "auto"
    ) -> tuple[SuperSample, BatchMetadata]:
        """A block-diagonal sparse super-problem of possibly differently sized sub-problems."""
        items = [self[int(i)] for i in indices]
        P_super = block_diag_csr(
            *(it[0].to_sparse_coo() if it[0].layout != torch.sparse_coo else it[0] for it in items)
        )
        A_super = block_diag_csr(
            *(it[2].to_sparse_coo() if it[2].layout != torch.sparse_coo else it[2] for it in items)
        )
        cat = lambda j: torch.cat([it[j] for it in items])  # noqa: E731
        metadata = BatchMetadata(
            B=len(items),
            n_pers=[it[1].shape[-1] for it in items],
            m_pers=[it[3].shape[-1] for it in items],
            soc_strategy=soc_strategy,
        )
        return (P_super, cat(1), A_super, cat(3), [it[4] for it in items], cat(5), cat(6), cat(7)), metadata

    def yield_super_batches(
        self, batch_size: int, shuffle: bool = True, soc_strategy: str = "auto"
    ) -> Iterable[tuple[SuperSample, BatchMetadata]]:
        indices = torch.randperm(len(self)) if shuffle else torch.arange(len(self))
        for i in range(0, len(self), batch_size):
            yield self.sample_super_batch(indices[i : i + batch_size].tolist(), soc_strategy=soc_strategy)
