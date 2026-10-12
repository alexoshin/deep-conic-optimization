"""Benchmark a trained policy vs. COSMO on held-out instances.

The learned policy is loaded from the checkpoints `best_policy.pt` and `best_initialization.pt`.
The baseline solver's hyperparameters are specified by the config.
Each instance is solved to `--eps-abs` tolerance on the ADMM residuals.
The solves are timed (synchronized on GPU) and failed solves are charged the iteration cap / time limit.
Results are saved to `<run_dir>/benchmark_<timestamp>/`, including full per-problem statistics and summaries.

Usage:
    dco-benchmark <run_dir> --dataset <test_set>
"""

import argparse
import csv
import json
import pathlib
import time

import numpy as np
import torch
import yaml

from ..config import TrainingConfig, policies
from ..data import ConeProblemDataset, problem_files
from ..solver import ProjectionProx, Solver, Solver_Status
from ..train import baseline_arm, build_module, learned_arm, load_checkpoints, setup_torch, solve_settings
from ..utils import format_time, resolve_path, timestamp

WARMUP_ITERATIONS = 50
RHO_KEYS = ("rho_eq_default", "rho_ineq_default", "rho_soc_default", "rho_psd_default", "alpha_default")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a trained policy vs. COSMO on held-out instances.")
    parser.add_argument("run_dir", help="training run directory (holds config.yaml and the checkpoints)")
    parser.add_argument("--dataset", required=True, help="held-out test dataset directory")
    parser.add_argument(
        "--arms", nargs="+", default=["baseline", "learned"], metavar="ARM",
        help="baseline, learned, constant (baseline's initial rho values), "
        "adaptive@T, learned@T (same arms under a specific rho-update gate T)",
    )  # fmt: skip
    parser.add_argument("--reference", metavar="ARM", help="arm the speedups are measured against (default: the first)")
    parser.add_argument("--eps-abs", type=float, nargs="+", default=[1e-3], help="absolute ADMM tolerances")
    parser.add_argument("--max-iter", type=int, default=10000, help="iteration limit per solve")
    parser.add_argument("--time-limit", type=float, help="wall-clock time limit per solve, in seconds (default: none)")
    parser.add_argument("--cg-eps-abs", type=float, help="override the forward CG absolute tolerance")
    parser.add_argument("--cg-eps-rel", type=float, help="override the forward CG relative tolerance")
    parser.add_argument("--n-repeats", type=int, default=1, help="timed solves per instance (median reported)")
    parser.add_argument("--n-warmup", type=int, default=2, help="untimed warm-up solves per arm before timing")
    parser.add_argument("--sgm-shift", type=float, default=1.0)
    parser.add_argument("--device", help="override the run's device")
    parser.add_argument("--out", help="output directory (default: <run_dir>/benchmark_<timestamp>)")
    return parser


def parse_arm(spec: str) -> tuple[str, float | None]:
    kind, _, tolerance = spec.partition("@")
    if kind not in ("baseline", "learned", "constant", "adaptive") or (kind in ("constant", "baseline") and tolerance):
        raise SystemExit(f"unknown arm {spec!r}: use baseline, learned, constant, adaptive@T or learned@T")
    if kind == "adaptive" and not tolerance:
        raise SystemExit(f"arm {spec!r} needs a rho-update gate: adaptive@T")
    return kind, (float(tolerance) if tolerance else None)


def build_arms(config: TrainingConfig, run_dir: pathlib.Path, specs: list[str], device, dtype) -> dict:
    """`{arm: (policy, initialization, adaptive_rho_tolerance)}`."""
    if len(set(specs)) != len(specs):
        raise SystemExit(f"repeated arm in {specs}")
    kinds = [parse_arm(spec) for spec in specs]
    learned = None
    if any(kind == "learned" for kind, _ in kinds):
        learned = learned_arm(config, device, dtype)
        load_checkpoints(run_dir, *learned, device)
    baseline = baseline_arm(config, device, dtype)
    arms = {}
    for spec, (kind, tolerance) in zip(specs, kinds):
        if kind == "baseline":
            arms[spec] = baseline
        elif kind == "adaptive":
            arms[spec] = (baseline[0], baseline[1], tolerance)
        elif kind == "constant":
            defaults = {key: getattr(baseline[0], key) for key in RHO_KEYS}
            arms[spec] = (
                build_module("policy.default", defaults, policies(), "policy", device, dtype),
                baseline[1],
                None,
            )
        else:
            arms[spec] = (*learned, config.adaptive_rho_tolerance if tolerance is None else tolerance)
    for policy, init, _ in arms.values():
        policy.eval()
        init.eval()
    return arms


def timed_solve(build, solve_kwargs: dict, device: torch.device, n_repeats: int) -> tuple:
    """Return `(result or exception, seconds per solve)`, with setup included."""
    times, result = [], None
    for _ in range(max(1, n_repeats)):
        solver = build()
        if device.type == "cuda":
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        t0 = time.perf_counter()
        try:
            result = solver.solve(**solve_kwargs)
        except (torch.linalg.LinAlgError, RuntimeError) as exc:
            return exc, [time.perf_counter() - t0]
        if device.type == "cuda":
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000.0)
        else:
            times.append(time.perf_counter() - t0)
    return result, times


def sgm(values: list[float], shift: float) -> float | None:
    """Shifted geometric mean `exp(mean(log(x + shift))) - shift`."""
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=float) + shift))) - shift) if values else None


def ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def charged(row: dict, max_iter: int, time_limit: float | None) -> tuple[float, float]:
    """Return `(iterations, seconds)` with a failed solve charged the caps."""
    if row["converged"]:
        return float(row["iterations"]), row["time_s"]
    return float(max_iter), (time_limit if time_limit is not None else row["time_s"])


def arm_statistics(arm_rows: list[dict], max_iter: int, time_limit: float | None, shift: float) -> dict:
    iters, secs = zip(*(charged(r, max_iter, time_limit) for r in arm_rows)) if arm_rows else ((), ())
    failed = [r["status"] for r in arm_rows if not r["converged"]]
    return {
        "solved": len(arm_rows) - len(failed),
        "total": len(arm_rows),
        "failures": {status: failed.count(status) for status in sorted(set(failed))},
        "iterations_median": float(np.median(iters)) if iters else None,
        "iterations_sgm": sgm(list(iters), shift),
        "time_median_s": float(np.median(secs)) if secs else None,
        "time_sgm_s": sgm(list(secs), shift),
    }


def summarize(
    rows: list[dict], eps_values: list[float], arms: list[str], reference: str, args
) -> tuple[dict, list[str]]:
    summary: dict = {"arms": arms, "reference": reference, "by_eps_abs": {}}
    lines = [f"failures charged {args.max_iter} iterations / {args.time_limit} s; reference arm {reference}"]
    for eps_abs in eps_values:
        key = f"{eps_abs:.0e}"
        by_arm = {name: [r for r in rows if r["eps_abs"] == eps_abs and r["arm"] == name] for name in arms}
        entry = {name: arm_statistics(by_arm[name], args.max_iter, args.time_limit, args.sgm_shift) for name in arms}
        lines.append(f"\neps_abs = {eps_abs:.1e}")
        for name in arms:
            st = entry[name]
            lines.append(
                f"  {name:>14}: solved {st['solved']}/{st['total']}, iterations median {st['iterations_median']:.0f} "
                f"(SGM {st['iterations_sgm']:.1f}), time median {st['time_median_s']:.4g} s (SGM {st['time_sgm_s']:.4g} s)"
                + (f", failures {st['failures']}" if st["failures"] else "")
            )
        entry["speedups"] = {}
        for name in arms:
            if name == reference:
                continue
            sp = {
                "iterations_median": ratio(entry[reference]["iterations_median"], entry[name]["iterations_median"]),
                "iterations_sgm": ratio(entry[reference]["iterations_sgm"], entry[name]["iterations_sgm"]),
                "time_median": ratio(entry[reference]["time_median_s"], entry[name]["time_median_s"]),
                "time_sgm": ratio(entry[reference]["time_sgm_s"], entry[name]["time_sgm_s"]),
            }
            entry["speedups"][name] = sp
            lines.append(
                f"  speedup {reference} / {name}: iterations median {sp['iterations_median']:.3g}x (SGM "
                f"{sp['iterations_sgm']:.3g}x), time median {sp['time_median']:.3g}x (SGM {sp['time_sgm']:.3g}x)"
            )
        summary["by_eps_abs"][key] = entry
    return summary, lines


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = resolve_path(args.run_dir)
    config = TrainingConfig.model_validate(yaml.safe_load((run_dir / "config.yaml").read_text()))
    if args.device:
        config = config.model_copy(update={"device": args.device})
    device, dtype = setup_torch(config)
    dataset_dir = resolve_path(args.dataset)
    paths = problem_files(dataset_dir)
    if not paths:
        raise SystemExit(f"no problem_*.pkl under {dataset_dir}")
    if dataset_dir.resolve() == resolve_path(config.dataset_path).resolve():
        raise SystemExit("the benchmark set is the training dataset; use a held-out set")
    instances = ConeProblemDataset(paths, gradient_method=config.gradient_method)
    arms = build_arms(config, run_dir, list(args.arms), device, dtype)
    names = list(arms)
    reference = args.reference or names[0]
    if reference not in arms:
        raise SystemExit(f"--reference {reference!r} is not one of {names}")
    # The forward pass solve settings only
    settings = solve_settings(config)
    for key in ("sigma", "cg_backward_eps_abs", "cg_backward_eps_rel", "cg_backward_max_iter", "adjoint_warm_start"):
        settings.pop(key)
    if args.cg_eps_abs is not None:
        settings["cg_eps_abs"] = args.cg_eps_abs
    if args.cg_eps_rel is not None:
        settings["cg_eps_rel"] = args.cg_eps_rel
    eps_values = sorted(args.eps_abs, reverse=True)
    out = pathlib.Path(args.out) if args.out else run_dir / f"benchmark_{timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"Run       : {run_dir}\nInstances : {len(paths)} from {dataset_dir}\nArms      : "
        + ", ".join(f"{name} (gate {arm[2]})" for name, arm in arms.items())
        + f"; reference {reference}\nOutput    : {out}\n"
    )

    problems = [tuple(instances[i][:5]) for i in range(len(instances))]

    def make_solver(problem, arm) -> Solver:
        P, q, A, b, cones = problem
        policy, initialization, _ = arm
        return Solver(
            P, q, A, b, prox=ProjectionProx(cones), policy=policy, initialization=initialization, sigma=config.sigma
        )

    def solve_kwargs(eps_abs: float, arm) -> dict:
        gate = {} if arm[2] is None else {"adaptive_rho_tolerance": float(arm[2])}
        return dict(
            max_iter=args.max_iter,
            eps_abs=eps_abs,
            eps_rel=0.0,
            eps_prim_inf=0.0,
            eps_dual_inf=0.0,
            check_termination=1,
            time_limit=args.time_limit,
            save_history=False,
            **settings,
            **gate,
        )

    with torch.no_grad():
        for arm in arms.values():
            for _ in range(args.n_warmup):
                make_solver(problems[0], arm).solve(
                    **{**solve_kwargs(eps_values[-1], arm), "max_iter": WARMUP_ITERATIONS}
                )

    rows: list[dict] = []
    width = max(14, *(len(name) + 6 for name in names))
    for eps_abs in eps_values:
        print(f"=== eps_abs = {eps_abs:.1e} " + "=" * 40)
        print(
            f"{'instance':>22}  "
            + "  ".join(f"{name + ' iters':>{width}}" for name in names)
            + "  "
            + "  ".join(f"{name + ' time':>{width}}" for name in names)
        )
        for path, problem in zip(paths, problems):
            row = {}
            for name in names:
                with torch.no_grad():
                    result, times = timed_solve(
                        lambda arm=arms[name]: make_solver(problem, arm),
                        solve_kwargs(eps_abs, arms[name]),
                        device,
                        args.n_repeats,
                    )
                if isinstance(result, BaseException):
                    row[name] = {
                        "status": f"EXCEPTION:{type(result).__name__}",
                        "converged": False,
                        "iterations": None,
                        "time_s": float(times[0]),
                    }
                else:
                    converged = result.status == Solver_Status.CONVERGED
                    row[name] = {
                        "status": result.status.name,
                        "converged": converged,
                        "iterations": result.num_iters if converged else None,
                        "time_s": float(np.median(times)),
                        "final_prim_res": float(result.prim_res_norm_hist[..., -1].max()),
                        "final_dual_res": float(result.dual_res_norm_hist[..., -1].max()),
                    }
                rows.append({"eps_abs": eps_abs, "instance": path.name, "arm": name, **row[name]})
            cell = lambda e: str(e["iterations"]) if e["converged"] else e["status"].lower()  # noqa: E731
            print(
                f"{path.stem:>22}  "
                + "  ".join(f"{cell(row[name]):>{width}}" for name in names)
                + "  "
                + "  ".join(f"{format_time(row[name]['time_s']):>{width}}" for name in names)
            )
        print()

    with (out / "per_problem.csv").open("w", newline="") as f:
        fields = [
            "eps_abs",
            "instance",
            "arm",
            "status",
            "converged",
            "iterations",
            "time_s",
            "final_prim_res",
            "final_dual_res",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary, lines = summarize(rows, eps_values, names, reference, args)
    summary["settings"] = {
        "run_dir": str(run_dir),
        "dataset": str(dataset_dir),
        "n_instances": len(paths),
        "eps_abs": eps_values,
        "max_iter": args.max_iter,
        "time_limit": args.time_limit,
        "n_repeats": args.n_repeats,
        "sgm_shift": args.sgm_shift,
        "device": str(device),
        "dtype": str(dtype),
        "adaptive_rho_tolerance": {name: arm[2] for name, arm in arms.items()},
        **settings,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
