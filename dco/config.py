"""Configuration schemas and registries of named components.

Dataset configs specify a problem generator and reference solver.
Training configs specify a policy, warm start, and unroll / training settings.
"""

import pathlib
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

LINSYS_METHODS = ("direct", "indirect")
PRECONDITIONERS = ("none", "jacobi")
GRADIENT_METHODS = ("autograd", "dk")
MODES = ("batched", "super_batched")
SOC_STRATEGIES = ("auto", "same_dim", "padded")
BEST_METRICS = ("loss", "final_prim_res", "final_dual_res", "final_res", "final_opt_gap")
REFERENCE_SOLVERS = ("cosmo", "clarabel", "clarabel_scp")


class DatasetConfig(BaseModel):
    """Config specifying how to generate a dataset of cone programs with reference solutions."""

    model_config = ConfigDict(extra="forbid")

    #: Output directory, relative to the package root unless absolute.
    save_path: str
    seed: int | None = None
    device: str = "cpu"
    precision: str = "double"
    n_problems: int = 100
    #: Key of :data:`GENERATORS`.
    problem_generator: str
    #: Generator keyword arguments sampled per instance: `{name: [low, high]}` draws uniformly.
    #: (integers when both bounds are integers); any other value is passed through unchanged.
    sampling: dict[str, Any] = {}
    #: Generator keyword arguments shared by every instance.
    fixed_kwargs: dict[str, Any] = {}
    #: Reference solver: `"clarabel"` (interior point), `"clarabel_scp"` (Clarabel reusing its factorization
    #: across the SCP iterations of a covariance-steering stream), or `"cosmo"` (torch solver using `policy`).
    solver: str = "clarabel"
    policy: str = "policy.adaptive_osqp"
    policy_kwargs: dict[str, Any] = {}
    solver_kwargs: dict[str, Any] = {}
    #: Apply ten Ruiz equilibration iterations to the data before solving and store the scaled data.
    equilibrate: bool = False
    #: Reject an instance whose reference dual exceeds this 2-norm (badly scaled instances).
    max_dual_norm: float | None = None

    @model_validator(mode="after")
    def check(self) -> "DatasetConfig":
        if self.solver not in REFERENCE_SOLVERS:
            raise ValueError(f"solver must be one of {REFERENCE_SOLVERS}, got {self.solver!r}")
        if self.precision not in ("double", "float"):
            raise ValueError(f"precision must be 'double' or 'float', got {self.precision!r}")
        return self


class TrainingConfig(BaseModel):
    """Config specifying how to train and evaluate a learned hyperparameter policy."""

    model_config = ConfigDict(extra="forbid")

    device: str = "cpu"
    precision: str = "double"
    seed: int | None = None
    #: Dataset directory (relative to the package root unless absolute), holding `problem_*.pkl`.
    dataset_path: str
    #: Training instances `sorted[:n_train_samples]` and validation instances `sorted[-n_val_samples:]`.
    #: The validation split selects the checkpoint, so the two must not overlap.
    n_train_samples: int
    n_val_samples: int
    #: Runs are written to `<output_dir>/<run_name>/<timestamp>/`; `run_name` defaults to the config's stem.
    output_dir: str = "outputs"
    run_name: str | None = None

    # Learned components (keys of POLICIES / INITIALIZATIONS)
    policy: str = "policy.feedback"
    policy_kwargs: dict[str, Any] = {}
    initialization: str = "init.default"
    initialization_kwargs: dict[str, Any] = {}

    # Baseline settings
    baseline_policy: str = "policy.adaptive_osqp"
    baseline_policy_kwargs: dict[str, Any] = {}
    baseline_initialization: str = "init.default"
    baseline_initialization_kwargs: dict[str, Any] = {}
    #: Rho-update gate: a new rho is adopted only when it moves by more than this factor.
    baseline_adaptive_rho_tolerance: float = 5.0

    # Unroll settings
    train_iters: int = 100
    val_iters: int | None = None
    #: Rho-update gate for the learned policy; 1.0 adopts the policy's rho on every iteration.
    adaptive_rho_tolerance: float = 1.0
    sigma: float = 1.0e-6
    linsys_solve_method: str = "indirect"
    cg_preconditioner: str = "jacobi"
    cg_eps_abs: float = 0.0
    cg_eps_rel: float = 1.0e-8
    cg_max_iter: int = 10000
    #: Adjoint CG settings of the implicit backward (`None` inherits the forward setting).
    cg_backward_eps_abs: float = 0.0
    cg_backward_eps_rel: float = 1.0e-6
    cg_backward_max_iter: int | None = None
    adjoint_warm_start: bool = True
    #: Backward of the PSD projection: `"dk"` (Daleckii-Krein) or `"autograd"` (through `eigh`).
    gradient_method: str = "dk"
    #: `"batched"` (dense batches of same-shaped problems) or `"super_batched"` (block-diagonal sparse).
    mode: str = "super_batched"
    soc_super_batch_strategy: str = "auto"

    # Training settings
    n_epochs: int = 100
    batch_size: int = 5
    learning_rate: float = 1.0e-3
    lr_schedule: str | None = None
    final_learning_rate: float | None = None
    eval_every_n_epochs: int = 1
    clip_grad_norm: float | None = None
    best_metric: str = "loss"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    #: Arguments of :func:`dco.loss.compute_loss`.
    loss_kwargs: dict[str, Any] = {}

    @model_validator(mode="after")
    def check(self) -> "TrainingConfig":
        checks = (
            (self.linsys_solve_method, LINSYS_METHODS, "linsys_solve_method"),
            (self.cg_preconditioner, PRECONDITIONERS, "cg_preconditioner"),
            (self.gradient_method, GRADIENT_METHODS, "gradient_method"),
            (self.mode, MODES, "mode"),
            (self.soc_super_batch_strategy, SOC_STRATEGIES, "soc_super_batch_strategy"),
            (self.best_metric, BEST_METRICS, "best_metric"),
            (self.precision, ("double", "float"), "precision"),
        )
        for value, allowed, name in checks:
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if self.lr_schedule not in (None, "cosine"):
            raise ValueError(f"lr_schedule must be null or 'cosine', got {self.lr_schedule!r}")
        return self


def load_yaml(path: str | pathlib.Path) -> dict:
    """The raw mapping of a YAML config file."""
    with pathlib.Path(path).open("r") as f:
        return yaml.safe_load(f) or {}


def load_dataset_config(path: str | pathlib.Path) -> DatasetConfig:
    return DatasetConfig.model_validate(load_yaml(path))


def load_training_config(path: str | pathlib.Path) -> TrainingConfig:
    return TrainingConfig.model_validate(load_yaml(path))


def policies() -> dict[str, type]:
    """Registry of hyperparameter policies, by config key."""
    from .policy import AdaptiveRhoOSQP, FeedbackPolicy, HyperparameterPolicy

    return {
        "policy.default": HyperparameterPolicy,
        "policy.adaptive_osqp": AdaptiveRhoOSQP,
        "policy.feedback": FeedbackPolicy,
    }


def initializations() -> dict[str, type]:
    """Registry of warm-start initializations, by config key."""
    from .warmstart import Initialization, RowWiseMLPWarmstart

    return {"init.default": Initialization, "init.ws_row_mlp": RowWiseMLPWarmstart}


def generators() -> dict[str, Any]:
    """Registry of problem generators, by config key."""
    from .problems import GENERATORS

    return GENERATORS


def lookup(registry: dict, key: str, what: str):
    if key not in registry:
        raise KeyError(f"unknown {what} {key!r}; available: {sorted(registry)}")
    return registry[key]
