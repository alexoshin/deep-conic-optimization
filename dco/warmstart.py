"""Initializations for the solver."""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from .cones import Cone, NonnegativeOrthant, PositiveSemidefiniteCone, SecondOrderCone, ZeroCone
from .linalg import inf_norm, safe_log
from .nn import MLP
from .super_batch import BatchMetadata

N_CONE_TYPES = 4
GLOBAL_CONTEXT_DIM = 4
N_VAR_BASE_FEATURES = 4  # Number of per-variable features
N_CON_BASE_FEATURES = 3 + N_CONE_TYPES  # Number of per-constraint features


def is_sparse(M: Tensor) -> bool:
    return M.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)


def cone_type_id(cone: Cone) -> int:
    if isinstance(cone, ZeroCone):
        return 0
    if isinstance(cone, NonnegativeOrthant):
        return 1
    if isinstance(cone, SecondOrderCone):
        return 2
    if isinstance(cone, PositiveSemidefiniteCone):
        return 3
    raise ValueError(f"Unknown cone type: {cone.__class__.__name__}")


def cone_type_onehot(cones: list[Cone], m: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    labels = torch.zeros(m, dtype=torch.long, device=device)
    for cone in cones:
        labels[cone.indices] = cone_type_id(cone)
    return F.one_hot(labels, num_classes=N_CONE_TYPES).to(dtype)


def matrix_diagonal(M: Tensor) -> Tensor:
    if is_sparse(M):
        Mc = M.to_sparse_coo().coalesce()
        idx, val = Mc.indices(), Mc.values()
        diag = torch.zeros(M.shape[-1], dtype=val.dtype, device=val.device)
        on_diag = idx[0] == idx[1]
        diag[idx[0][on_diag]] = val[on_diag]
        return diag
    return torch.diagonal(M)


def row_col_stats(M: Tensor, m: int, n: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes `(row_l2, row_nnz, col_l2, col_nnz)` of an `(m, n)` matrix."""
    if is_sparse(M):
        Mc = M.to_sparse_coo().coalesce()
        rows, cols = Mc.indices()
        val = Mc.values()
        kwargs = {"dtype": val.dtype, "device": val.device}
        row_l2 = torch.zeros(m, **kwargs).scatter_add_(0, rows, val * val).sqrt()
        col_l2 = torch.zeros(n, **kwargs).scatter_add_(0, cols, val * val).sqrt()
        ones = torch.ones_like(val)
        row_nnz = torch.zeros(m, **kwargs).scatter_add_(0, rows, ones)
        col_nnz = torch.zeros(n, **kwargs).scatter_add_(0, cols, ones)
        return row_l2, row_nnz, col_l2, col_nnz
    nz = (M != 0).to(M.dtype)
    return (
        torch.linalg.vector_norm(M, ord=2, dim=-1),
        nz.sum(dim=-1),
        torch.linalg.vector_norm(M, ord=2, dim=-2),
        nz.sum(dim=-2),
    )


def global_context(q: Tensor, b: Tensor, batch_metadata: BatchMetadata | None) -> tuple[Tensor, Tensor]:
    """Compute per-problem statistics, broadcast onto each element."""
    if batch_metadata is not None:
        reducer_n, reducer_m = batch_metadata.reducers(q.device)
        q_inf, b_inf = reducer_n.inf_norm(q), reducer_m.inf_norm(b)
        n_per = torch.tensor(batch_metadata.n_pers, dtype=q.dtype, device=q.device)
        m_per = torch.tensor(batch_metadata.m_pers, dtype=q.dtype, device=q.device)
        ctx = torch.stack(
            [safe_log(1.0 + q_inf), safe_log(1.0 + b_inf), safe_log(1.0 + n_per), safe_log(1.0 + m_per)], dim=-1
        )
        return reducer_n.broadcast(ctx.T).T, reducer_m.broadcast(ctx.T).T
    n, m = q.shape[-1], b.shape[-1]
    q_inf, b_inf = inf_norm(q), inf_norm(b)
    ctx = torch.stack(
        [
            safe_log(1.0 + q_inf),
            safe_log(1.0 + b_inf),
            safe_log(1.0 + torch.full_like(q_inf, float(n))),
            safe_log(1.0 + torch.full_like(b_inf, float(m))),
        ],
        dim=-1,
    )
    return ctx.expand(n, GLOBAL_CONTEXT_DIM), ctx.expand(m, GLOBAL_CONTEXT_DIM)


def variable_constraint_features(
    P: Tensor,
    q: Tensor,
    A: Tensor,
    b: Tensor,
    cones: list[Cone],
    batch_metadata: BatchMetadata | None,
    use_global_context: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute the variable and constraint features."""
    n, m = q.shape[-1], b.shape[-1]
    row_l2, row_nnz, col_l2, col_nnz = row_col_stats(A, m, n)
    var_feats = torch.stack([q, matrix_diagonal(P), col_l2, col_nnz], dim=-1)
    con_feats = torch.cat(
        [torch.stack([b, row_l2, row_nnz], dim=-1), cone_type_onehot(cones, m, b.device, b.dtype)], dim=-1
    )
    if use_global_context:
        ctx_n, ctx_m = global_context(q, b, batch_metadata)
        var_feats = torch.cat([var_feats, ctx_n], dim=-1)
        con_feats = torch.cat([con_feats, ctx_m], dim=-1)
    return var_feats, con_feats


class Initialization(nn.Module):
    """The zero start `x0 = z0 = y0 = 0`."""

    def forward(
        self,
        P: Float[Tensor, "... n n"],
        q: Float[Tensor, "... n"],
        A: Float[Tensor, "... m n"],
        b: Float[Tensor, "... m"],
        blocks: list[Cone],
        batch_metadata: BatchMetadata | None = None,
    ) -> tuple[Float[Tensor, "... n"], Float[Tensor, "... m"], Float[Tensor, "... m"]]:
        z = torch.zeros_like(b)
        return torch.zeros_like(q), z, torch.zeros_like(z)


class RowWiseMLPWarmstart(Initialization):
    """Learned warm start."""

    def __init__(
        self,
        hidden_sizes: list[int] = [64, 64],
        activation_fn: nn.Module = nn.ReLU(),
        use_global_context: bool = True,
    ) -> None:
        super().__init__()
        self.use_global_context = use_global_context
        extra = GLOBAL_CONTEXT_DIM if use_global_context else 0
        self.x_net = MLP(
            N_VAR_BASE_FEATURES + extra, hidden_sizes, 1, activation_fn=activation_fn, zero_last_layer=True
        )
        self.zy_net = MLP(
            N_CON_BASE_FEATURES + extra, hidden_sizes, 2, activation_fn=activation_fn, zero_last_layer=True
        )

    def forward(self, P, q, A, b, blocks, batch_metadata=None):
        var_feats, con_feats = variable_constraint_features(P, q, A, b, blocks, batch_metadata, self.use_global_context)
        x = self.x_net(var_feats).squeeze(-1)
        zy = self.zy_net(con_feats)
        return x, zy[..., 0], zy[..., 1]
