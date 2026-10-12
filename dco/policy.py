"""Hyperparameter policies."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jaxtyping import Float
import torch
from torch import Tensor, nn

from .cones import (
    Cone,
    NonnegativeOrthant,
    PaddedBatchedSecondOrderCone,
    PositiveSemidefiniteCone,
    SecondOrderCone,
    ZeroCone,
)
from .linalg import inf_norm, safe_log
from .nn import MLP
from .super_batch import BatchMetadata

if TYPE_CHECKING:
    from .functional import Solver_Residuals

# These cones require a single rho for the entire block
BLOCK_COUPLED_TYPES = (SecondOrderCone, PositiveSemidefiniteCone)


@dataclass
class PolicyState:
    """The per-iteration policy state carried through the ADMM loop."""

    rhos: Float[Tensor, "... m"]
    alpha: Float[Tensor, "..."]


def block_cone_family(block: Cone) -> str:
    """`"eq"`, `"ineq"`, `"soc"` or `"psd"`."""
    if isinstance(block, ZeroCone):
        return "eq"
    if isinstance(block, NonnegativeOrthant):
        return "ineq"
    if isinstance(block, SecondOrderCone):
        return "soc"
    if isinstance(block, PositiveSemidefiniteCone):
        return "psd"
    raise ValueError(f"Unknown block type: {block.__class__.__name__}")


class HyperparameterPolicy(nn.Module):
    """Base hyperparameter policy class.

    Defaults to a fixed `rho` per cone family and fixed `alpha`.
    """

    def __init__(
        self,
        rho_eq_default: float = 100.0,
        rho_ineq_default: float = 0.1,
        rho_soc_default: float = 0.1,
        rho_psd_default: float = 0.1,
        alpha_default: float = 1.6,
    ) -> None:
        super().__init__()
        self.rho_eq_default = rho_eq_default
        self.rho_ineq_default = rho_ineq_default
        self.rho_soc_default = rho_soc_default
        self.rho_psd_default = rho_psd_default
        self.alpha_default = alpha_default
        # Will be set after calling `initialize`
        self.blocks: list[Cone] = []  # a plain list so the blocks are not registered as submodules
        self.batch_metadata: BatchMetadata | None = None

    def rho_default_for_block(self, block: Cone) -> float:
        return getattr(self, f"rho_{block_cone_family(block)}_default")

    def initialize(
        self,
        P: Float[Tensor, "... n n"],
        q: Float[Tensor, "... n"],
        A: Float[Tensor, "... m n"],
        b: Float[Tensor, "... m"],
        blocks: list[Cone],
        batch_metadata: BatchMetadata | None = None,
    ) -> PolicyState:
        """Initialize the policy state using the default values of `rho` and `alpha`."""
        self.blocks = blocks
        self.batch_metadata = batch_metadata
        tensor_kwargs = {"dtype": b.dtype, "device": b.device}
        rhos = torch.zeros(b.shape[-1], **tensor_kwargs)
        for block in blocks:
            rhos.index_fill_(-1, block.indices, self.rho_default_for_block(block))
        return PolicyState(rhos, self.alpha_state(self.alpha_default, batch_metadata, tensor_kwargs))

    def rectify_rhos(self, rhos: Float[Tensor, "... m"]) -> Float[Tensor, "... m"]:
        """Average the rhos within each SOC / PSD block and broadcast the mean back over the block.

        The scaled projection of a second-order or PSD block couples all of its rows,
        so the block needs a single shared rho for the projection to be well defined.
        """
        rhos = rhos.clone()
        for block in self.blocks:
            if isinstance(block, PaddedBatchedSecondOrderCone):
                sub = rhos[..., block.padded_indices]
                counts = block.mask.sum(dim=-1, keepdim=True)
                means = (sub * block.mask).sum(dim=-1, keepdim=True) / counts
                rhos[..., block.indices] = means.expand_as(sub)[..., block.valid_row, block.valid_col]
            elif not isinstance(block, BLOCK_COUPLED_TYPES):
                continue
            elif getattr(block, "indices_2d", None) is not None:
                sub = rhos[..., block.indices_2d]
                rhos[..., block.indices_2d] = sub.mean(dim=-1, keepdim=True).expand_as(sub)
            else:
                sub = rhos[..., block.indices]
                rhos[..., block.indices] = sub.mean(dim=-1, keepdim=True).expand_as(sub)
        return rhos

    @staticmethod
    def alpha_state(alpha: float | Tensor, batch_metadata: BatchMetadata | None, tensor_kwargs: dict) -> Tensor:
        """Return the per-sub-problem `(B,)` alpha, or a torch scalar when not super-batched."""
        shape = () if batch_metadata is None else (batch_metadata.B,)
        return alpha * torch.ones(shape, **tensor_kwargs)

    def seg_inf_norm(self, v: Tensor, space: str) -> Tensor:
        """Compute the per-problem infinity norm of a flat super-vector (`(B,)`), or the trailing-dim norm."""
        if self.batch_metadata is None:
            return inf_norm(v)
        reducer_n, reducer_m = self.batch_metadata.reducers(v.device)
        return (reducer_n if space == "n" else reducer_m).inf_norm(v)

    def expand_per_problem(self, per_problem: Tensor, like: Tensor, space: str) -> Tensor:
        """Broadcast a per-problem scalar back over each sub-problem's segment of `like`."""
        if self.batch_metadata is None:
            return per_problem.unsqueeze(-1).expand_as(like)
        reducer_n, reducer_m = self.batch_metadata.reducers(like.device)
        return (reducer_n if space == "n" else reducer_m).broadcast(per_problem)

    def forward(
        self,
        x: Float[Tensor, "... n"],
        z: Float[Tensor, "... m"],
        y: Float[Tensor, "... m"],
        residuals: "Solver_Residuals",
        k: int,
        state: PolicyState,
    ) -> tuple[Float[Tensor, "... m"], Float[Tensor, "..."], PolicyState]:
        return state.rhos, state.alpha, state


class AdaptiveRhoOSQP(HyperparameterPolicy):
    """OSQP / COSMO residual-balancing rule: `rho <- rho sqrt(r_prim / r_dual)` in normalized residuals."""

    def forward(self, x, z, y, residuals, k, state):
        if k == 0:
            return state.rhos, state.alpha, state
        dual_norm = self.seg_inf_norm(residuals.dual_res, "n") / (residuals.dual_rel_norm + 1e-12)
        prim_norm = self.seg_inf_norm(residuals.prim_res, "m") / (residuals.prim_rel_norm + 1e-12)
        sqrt_r_ratio = torch.sqrt(prim_norm / (dual_norm + 1e-12))
        rhos_new = self.expand_per_problem(sqrt_r_ratio, state.rhos, "m") * state.rhos
        return rhos_new, state.alpha, PolicyState(rhos_new, state.alpha)


class FeedbackPolicy(HyperparameterPolicy):
    """Learned feedback policy.

    Two small MLPs predict increments to the logits of `rho` and `alpha` based on the current residuals.
    The logits are squashed into the ranges `[rho_min, rho_max]` (log-uniformly) and `(0, 2)`.
    The update is recurrent in logit space: `logit_rho^{k+1} = logit_rho^k + MLP_rho(residuals^k)` and similarly for `alpha`.
    The initial logits are also learnable parameters, so the policy can learn a fixed per-family rho as well.

    The output layers are zero-initialized, so the untrained policy reproduces fixed-hyperparameter COSMO exactly.

    With `rectify_rhos_per_cone` the per-row rho is averaged within each SOC / PSD block.
    """

    def __init__(
        self,
        rho_eq_default: float = 100.0,
        rho_ineq_default: float = 0.1,
        rho_soc_default: float = 0.1,
        rho_psd_default: float = 0.1,
        alpha_default: float = 1.6,
        hidden_sizes: list[int] = [32, 32],
        activation_fn: nn.Module = nn.ReLU(),
        rho_min: float = 1e-4,
        rho_max: float = 1e4,
        rectify_rhos_per_cone: bool = False,
    ) -> None:
        super().__init__(rho_eq_default, rho_ineq_default, rho_soc_default, rho_psd_default, alpha_default)
        self.rectify_rhos_per_cone = rectify_rhos_per_cone
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.log_rho_min = torch.log(torch.tensor(rho_min))
        self.log_rho_max = torch.log(torch.tensor(rho_max))

        self.logit_rho_eq = nn.Parameter(self.rho_to_logit(torch.tensor(rho_eq_default)))
        self.logit_rho_ineq = nn.Parameter(self.rho_to_logit(torch.tensor(rho_ineq_default)))
        self.logit_rho_soc = nn.Parameter(self.rho_to_logit(torch.tensor(rho_soc_default)))
        self.logit_rho_psd = nn.Parameter(self.rho_to_logit(torch.tensor(rho_psd_default)))
        self.logit_alpha = nn.Parameter(self.alpha_to_logit(torch.tensor(alpha_default)))

        self.n_rho_inputs = 7
        self.rho_policy = MLP(self.n_rho_inputs, hidden_sizes, 1, activation_fn=activation_fn, zero_last_layer=True)
        self.n_alpha_inputs = 5
        self.alpha_policy = MLP(self.n_alpha_inputs, hidden_sizes, 1, activation_fn=activation_fn, zero_last_layer=True)

    def logit_to_rho(self, logit: Tensor) -> Tensor:
        return torch.exp(self.log_rho_min + (self.log_rho_max - self.log_rho_min) * torch.sigmoid(logit))

    def rho_to_logit(self, rho: Tensor) -> Tensor:
        return torch.logit((torch.log(rho) - self.log_rho_min) / (self.log_rho_max - self.log_rho_min), eps=1e-12)

    def logit_to_alpha(self, logit: Tensor) -> Tensor:
        return 2.0 * torch.sigmoid(logit)

    def alpha_to_logit(self, alpha: Tensor) -> Tensor:
        return torch.logit(0.5 * alpha)

    def logit_rho_for_block(self, block: Cone) -> Tensor:
        return getattr(self, f"logit_rho_{block_cone_family(block)}")

    def initialize(self, P, q, A, b, blocks, batch_metadata=None) -> PolicyState:
        self.blocks = blocks
        self.batch_metadata = batch_metadata
        tensor_kwargs = {"dtype": b.dtype, "device": b.device}
        rhos = torch.zeros(b.shape[-1], **tensor_kwargs)
        for block in blocks:
            rhos = rhos.clone()
            rhos[..., block.indices] = self.logit_to_rho(self.logit_rho_for_block(block))
        alpha = self.alpha_state(self.logit_to_alpha(self.logit_alpha), batch_metadata, tensor_kwargs)
        return PolicyState(rhos=rhos, alpha=alpha)

    def forward(self, x, z, y, residuals, k, state):
        """Predict `(rho, alpha)` from the current state."""

        # fmt: off
        logit_rhos_prev = self.rho_to_logit(state.rhos)
        logit_alpha_prev = self.alpha_to_logit(state.alpha)

        prim_normalized = residuals.prim_res / (self.expand_per_problem(residuals.prim_rel_norm, residuals.prim_res, "m") + 1e-12)
        dual_normalized = residuals.dual_res / (self.expand_per_problem(residuals.dual_rel_norm, residuals.dual_res, "n") + 1e-12)
        prim_norm = self.seg_inf_norm(prim_normalized, "m")
        dual_norm = self.seg_inf_norm(dual_normalized, "n")
        admm_prim = residuals.admm_prim_z
        admm_dual = residuals.admm_dual_z
        admm_prim_norm = self.seg_inf_norm(admm_prim, "m")
        admm_dual_norm = self.seg_inf_norm(admm_dual, "m")

        rho_features = torch.stack([
            logit_rhos_prev,
            z,
            y,
            prim_normalized,
            safe_log(self.expand_per_problem(dual_norm, z, "m")),
            admm_prim,
            admm_dual,
        ], dim=-1)
        alpha_features = torch.stack([
            logit_alpha_prev,
            safe_log(prim_norm),
            safe_log(dual_norm),
            safe_log(admm_prim_norm),
            safe_log(admm_dual_norm),
        ], dim=-1)

        rhos = self.logit_to_rho(logit_rhos_prev + self.rho_policy(rho_features).squeeze(-1))
        if self.rectify_rhos_per_cone:
            rhos = self.rectify_rhos(rhos)
        alpha = self.logit_to_alpha(logit_alpha_prev + self.alpha_policy(alpha_features).squeeze(-1))
        # fmt: on

        return rhos, alpha, PolicyState(rhos=rhos, alpha=alpha)
