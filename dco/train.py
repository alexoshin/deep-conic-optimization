"""Unrolled solver training loop."""

import json
import logging
import math
import pathlib
import shutil
import sys
import time

import numpy as np
import torch

from .config import TrainingConfig, initializations, lookup, policies
from .data import ConeProblemDataset, problem_files
from .loss import compute_loss
from .nn import count_parameters
from .policy import HyperparameterPolicy
from .utils import resolve_path, timestamp
from .warmstart import Initialization


def setup_torch(config: TrainingConfig) -> tuple[torch.device, torch.dtype]:
    """Get the default device and dtype from the config."""
    device = torch.device(config.device)
    dtype = torch.float64 if config.precision == "double" else torch.float32
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    return device, dtype


def build_module(key: str, kwargs: dict, registry: dict, what: str, device: torch.device, dtype: torch.dtype):
    return lookup(registry, key, what)(**kwargs).to(device=device, dtype=dtype)


def learned_arm(config: TrainingConfig, device, dtype) -> tuple[HyperparameterPolicy, Initialization]:
    policy = build_module(config.policy, config.policy_kwargs, policies(), "policy", device, dtype)
    init = build_module(
        config.initialization, config.initialization_kwargs, initializations(), "initialization", device, dtype
    )
    return policy, init


def baseline_arm(config: TrainingConfig, device, dtype) -> tuple[HyperparameterPolicy, Initialization, float]:
    policy = build_module(config.baseline_policy, config.baseline_policy_kwargs, policies(), "policy", device, dtype)
    init = build_module(
        config.baseline_initialization,
        config.baseline_initialization_kwargs,
        initializations(),
        "initialization",
        device,
        dtype,
    )
    return policy, init, config.baseline_adaptive_rho_tolerance


def solve_settings(config: TrainingConfig) -> dict:
    """The solver settings for an unroll, used by :func:`dco.loss.compute_loss`."""
    return dict(
        sigma=config.sigma,
        linsys_solve_method=config.linsys_solve_method,
        cg_preconditioner=config.cg_preconditioner,
        cg_max_iter=config.cg_max_iter,
        cg_eps_abs=config.cg_eps_abs,
        cg_eps_rel=config.cg_eps_rel,
        cg_backward_eps_abs=config.cg_backward_eps_abs,
        cg_backward_eps_rel=config.cg_backward_eps_rel,
        cg_backward_max_iter=config.cg_backward_max_iter,
        adjoint_warm_start=config.adjoint_warm_start,
    )


def load_checkpoints(run_dir: pathlib.Path, policy: torch.nn.Module, initialization: torch.nn.Module, device) -> None:
    """Load the best checkpoint."""
    for module, name in ((policy, "best_policy.pt"), (initialization, "best_initialization.pt")):
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: the run has no selected checkpoint")
        module.load_state_dict(torch.load(path, map_location=device, weights_only=True))


def median_iqr(stacked: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qs = torch.tensor([0.25, 0.5, 0.75], dtype=stacked.dtype, device=stacked.device)
    q25, median, q75 = torch.quantile(stacked, qs, dim=0).detach().cpu().numpy()
    return median, q25, q75


def run_epoch(
    dataset: ConeProblemDataset,
    batch_size: int,
    policy: HyperparameterPolicy,
    n_iters: int,
    initialization: Initialization,
    opt: torch.optim.Optimizer | None = None,
    trainable_params: list | None = None,
    clip_grad_norm: float | None = None,
    mode: str = "super_batched",
    soc_super_batch_strategy: str = "auto",
    **loss_kwargs,
) -> dict:
    """Run one pass over the `dataset` (a training step per batch when `opt` is given, evaluation otherwise).

    Returns the mean loss and the median + IQR residual and optimality-gap histories.
    """
    if mode == "super_batched":
        batches = dataset.yield_super_batches(batch_size, soc_strategy=soc_super_batch_strategy)
    elif mode == "batched":
        batches = ((batch, None) for batch in dataset.yield_batches(batch_size))
    else:
        raise ValueError(f"mode must be 'batched' or 'super_batched', got {mode!r}")

    losses, dual_hists, prim_hists, gap_hists = [], [], [], []
    for batch_data, batch_metadata in batches:
        loss, result, opt_gap_hist = compute_loss(
            batch_data, policy, n_iters, initialization=initialization, batch_metadata=batch_metadata, **loss_kwargs
        )
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, clip_grad_norm)
            opt.step()
        losses.append(loss.detach())
        dual_hists.append(result.dual_res_norm_hist.detach())
        prim_hists.append(result.prim_res_norm_hist.detach())
        gap_hists.append(opt_gap_hist.detach())
    return {
        "loss": torch.stack(losses).mean().item(),
        "dual": median_iqr(torch.cat(dual_hists)),
        "prim": median_iqr(torch.cat(prim_hists)),
        "gap": median_iqr(torch.cat(gap_hists)),
    }


def selection_value(result: dict, best_metric: str) -> float:
    if best_metric == "loss":
        return result["loss"]
    if best_metric == "final_opt_gap":
        return float(result["gap"][0][-1])
    if best_metric == "final_prim_res":
        return float(result["prim"][0][-1])
    if best_metric == "final_dual_res":
        return float(result["dual"][0][-1])
    if best_metric == "final_res":
        return max(float(result["prim"][0][-1]), float(result["dual"][0][-1]))
    raise ValueError(f"Unknown best_metric: {best_metric}")


def result_str(r: dict) -> str:
    return f"loss {r['loss']:.3e}, final dual res {r['dual'][0][-1]:.2e}, prim res {r['prim'][0][-1]:.2e}, opt gap {r['gap'][0][-1]:.2e}"


def make_run_dir(config: TrainingConfig, config_path: pathlib.Path) -> pathlib.Path:
    name = config.run_name or config_path.stem
    run_dir = resolve_path(config.output_dir) / name / timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    return run_dir


def make_logger(run_dir: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger(str(run_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(run_dir / "log.txt"))
    return logger


def train(config_path: str | pathlib.Path) -> pathlib.Path:
    """Run the training specified by `config_path`."""
    config_path = pathlib.Path(config_path)
    config = TrainingConfig.model_validate(__import__("yaml").safe_load(config_path.read_text()))
    device, dtype = setup_torch(config)
    run_dir = make_run_dir(config, config_path)
    log = make_logger(run_dir).info
    log(f"Run directory: {run_dir}")

    dataset_path = resolve_path(config.dataset_path)
    files = problem_files(dataset_path)
    if not files:
        raise FileNotFoundError(f"no problem_*.pkl under {dataset_path}")
    train_paths, val_paths = files[: config.n_train_samples], files[-config.n_val_samples :]
    if set(train_paths) & set(val_paths):
        raise ValueError("the training and validation splits overlap; lower n_train_samples or n_val_samples")
    train_dataset = ConeProblemDataset(train_paths, gradient_method=config.gradient_method)
    val_dataset = ConeProblemDataset(val_paths, gradient_method=config.gradient_method)
    log(f"Loaded {len(train_dataset)} train / {len(val_dataset)} validation problems from {dataset_path}")

    policy, initialization = learned_arm(config, device, dtype)
    log(
        f"{policy.__class__.__name__}: {count_parameters(policy)} parameters; "
        f"{initialization.__class__.__name__}: {count_parameters(initialization)} parameters"
    )
    baseline_policy, baseline_init, baseline_tolerance = baseline_arm(config, device, dtype)

    val_iters = config.val_iters if config.val_iters is not None else config.train_iters
    epoch_kwargs = dict(
        mode=config.mode,
        soc_super_batch_strategy=config.soc_super_batch_strategy,
        adaptive_rho_tolerance=config.adaptive_rho_tolerance,
        **solve_settings(config),
        **config.loss_kwargs,
    )
    baseline_kwargs = {**epoch_kwargs, "adaptive_rho_tolerance": baseline_tolerance}

    params = list(policy.parameters()) + list(initialization.parameters())
    opt = torch.optim.Adam(params, lr=config.learning_rate)
    scheduler = None
    if config.lr_schedule == "cosine":
        eta_min = config.final_learning_rate if config.final_learning_rate is not None else config.learning_rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.n_epochs, eta_min=eta_min)

    with torch.no_grad():
        baseline = run_epoch(
            val_dataset, config.batch_size, baseline_policy, val_iters, baseline_init, **baseline_kwargs
        )
        val_result = run_epoch(val_dataset, config.batch_size, policy, val_iters, initialization, **epoch_kwargs)
    log(f"Baseline ({baseline_policy.__class__.__name__}, gate {baseline_tolerance}) - {result_str(baseline)}")
    log(f"Initial evaluation - {result_str(val_result)}")

    best_policy_path, best_init_path = run_dir / "best_policy.pt", run_dir / "best_initialization.pt"
    current_best = selection_value(val_result, config.best_metric)
    best_epoch = 0
    torch.save(policy.state_dict(), best_policy_path)
    torch.save(initialization.state_dict(), best_init_path)

    history = {"train_loss": [], "val_loss": [val_result["loss"]], "val_epochs": [0]}
    evals_since_improvement = 0
    training_start = time.time()
    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()
        train_result = run_epoch(
            train_dataset,
            config.batch_size,
            policy,
            config.train_iters,
            initialization,
            opt=opt,
            trainable_params=params,
            clip_grad_norm=config.clip_grad_norm,
            **epoch_kwargs,
        )
        history["train_loss"].append(train_result["loss"])
        torch.save(policy.state_dict(), run_dir / "policy.pt")
        torch.save(initialization.state_dict(), run_dir / "initialization.pt")
        log(f"Epoch {epoch}/{config.n_epochs} - {result_str(train_result)}, {time.time() - t0:.1f} s")
        if scheduler is not None:
            scheduler.step()

        if epoch % config.eval_every_n_epochs == 0:
            with torch.no_grad():
                val_result = run_epoch(
                    val_dataset, config.batch_size, policy, val_iters, initialization, **epoch_kwargs
                )
            history["val_loss"].append(val_result["loss"])
            history["val_epochs"].append(epoch)
            value = selection_value(val_result, config.best_metric)
            margin = config.early_stopping_min_delta * abs(current_best) if math.isfinite(current_best) else 0.0
            improved = value < current_best - margin
            if value < current_best:
                current_best, best_epoch = value, epoch
                torch.save(policy.state_dict(), best_policy_path)
                torch.save(initialization.state_dict(), best_init_path)
            evals_since_improvement = 0 if improved else evals_since_improvement + 1
            log(f"Evaluation - {result_str(val_result)}" + (" (new best)" if improved else ""))
            if config.early_stopping_patience is not None and evals_since_improvement >= config.early_stopping_patience:
                log(
                    f"Early stopping at epoch {epoch}: no improvement in {config.best_metric} for "
                    f"{evals_since_improvement} evaluations (best {current_best:.3e} at epoch {best_epoch})"
                )
                break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    training_time = time.time() - training_start
    load_checkpoints(run_dir, policy, initialization, device)
    with torch.no_grad():
        val_result = run_epoch(val_dataset, config.batch_size, policy, val_iters, initialization, **epoch_kwargs)
    log(
        f"Finished training in {training_time / 60:.1f} min. Best model (epoch {best_epoch}) - {result_str(val_result)}"
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    summary = {
        "best_metric": config.best_metric,
        "best_value": current_best,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "training_time_s": training_time,
        "val_loss": val_result["loss"],
        "val_final_dual_res_median": float(val_result["dual"][0][-1]),
        "val_final_prim_res_median": float(val_result["prim"][0][-1]),
        "val_final_opt_gap_median": float(val_result["gap"][0][-1]),
        "baseline_val_loss": baseline["loss"],
        "baseline_final_dual_res_median": float(baseline["dual"][0][-1]),
        "baseline_final_prim_res_median": float(baseline["prim"][0][-1]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return run_dir
