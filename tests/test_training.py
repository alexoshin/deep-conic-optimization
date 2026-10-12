"""End-to-end test on a tiny max-cut dataset."""

import json
import pathlib

import yaml

from dco.cli import benchmark, generate
from dco.train import train


def write_yaml(path: pathlib.Path, data: dict) -> pathlib.Path:
    path.write_text(yaml.safe_dump(data))
    return path


def dataset_config(tmp_path: pathlib.Path, name: str, seed: int, n_problems: int) -> pathlib.Path:
    return write_yaml(
        tmp_path / f"{name}.yaml",
        {
            "save_path": str(tmp_path / "data" / name),
            "seed": seed,
            "device": "cpu",
            "precision": "double",
            "n_problems": n_problems,
            "problem_generator": "max_cut",
            "fixed_kwargs": {"n": 6, "edge_density": 0.5, "layout": "dense"},
            "solver": "clarabel",
            "solver_kwargs": {"max_iter": 200, "conv_tol": 1.0e-9, "verbose": False},
        },
    )


def test_generate_train_benchmark(tmp_path):
    generate.main([str(dataset_config(tmp_path, "train", 0, 6))])
    generate.main([str(dataset_config(tmp_path, "test", 1, 3))])
    assert len(list((tmp_path / "data" / "train").glob("problem_*.pkl"))) == 6

    config = write_yaml(
        tmp_path / "smoke.yaml",
        {
            "device": "cpu",
            "precision": "double",
            "seed": 0,
            "dataset_path": str(tmp_path / "data" / "train"),
            "n_train_samples": 4,
            "n_val_samples": 2,
            "output_dir": str(tmp_path / "outputs"),
            "policy": "policy.feedback",
            "policy_kwargs": {
                "hidden_sizes": [8, 8],
                "rho_min": 1.0e-3,
                "rho_max": 1.0e3,
                "rectify_rhos_per_cone": True,
            },
            "initialization": "init.ws_row_mlp",
            "initialization_kwargs": {"hidden_sizes": [8, 8]},
            "train_iters": 10,
            "val_iters": 10,
            "linsys_solve_method": "indirect",
            "cg_preconditioner": "jacobi",
            "mode": "super_batched",
            "n_epochs": 2,
            "batch_size": 2,
            "learning_rate": 1.0e-3,
            "lr_schedule": "cosine",
            "final_learning_rate": 1.0e-4,
            "eval_every_n_epochs": 1,
            "clip_grad_norm": 1.0,
            "best_metric": "loss",
            "early_stopping_patience": 5,
            "loss_kwargs": {
                "metric": "residuals_sq",
                "all_iterates": True,
                "normalization": "initial_gap",
                "transform": "log",
            },
            "baseline_policy": "policy.adaptive_osqp",
            "baseline_adaptive_rho_tolerance": 5.0,
        },
    )
    run_dir = train(config)
    for name in ("config.yaml", "best_policy.pt", "best_initialization.pt", "history.json", "summary.json", "log.txt"):
        assert (run_dir / name).is_file(), name
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["epochs_run"] == 2 and summary["best_metric"] == "loss"
    history = json.loads((run_dir / "history.json").read_text())
    assert len(history["train_loss"]) == 2 and len(history["val_loss"]) == 3

    benchmark.main(
        [
            str(run_dir),
            "--dataset",
            str(tmp_path / "data" / "test"),
            "--eps-abs",
            "1e-3",
            "--max-iter",
            "2000",
            "--n-warmup",
            "1",
        ]
    )
    out = sorted(run_dir.glob("benchmark_*"))[-1]
    result = json.loads((out / "summary.json").read_text())
    stats = result["by_eps_abs"]["1e-03"]
    assert stats["baseline"]["total"] == 3 and stats["learned"]["total"] == 3
    assert stats["baseline"]["solved"] == 3
    assert (out / "per_problem.csv").is_file()
