"""Generate a dataset of cone programs with reference solutions.

Each instance is drawn from the generator specified by the config, solved by the reference solver,
then saved as a pickle file `problem_{i:05d}.pkl` containing `(P, q, A, b, cones, result)`.
Failed solves are resampled.

The covariance-steering generators run an SCP loop per instance.
Each subproblem is saved as `problem_{i:05d}_iter_{j:02d}.pkl`.

Usage:
    dco-generate <config_path>
"""

import argparse
import json
import pathlib
import pickle

from jaxtyping import Float
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from ..clarabel import make_solver_result, solve_with_stats
from ..cones import Cone
from ..config import DatasetConfig, generators, load_dataset_config, lookup, policies
from ..equilibration import equilibriate_
from ..solver import ProjectionProx, Solver, Solver_Result
from ..utils import resolve_path


SCP_GENERATORS = ("cs_double_integrator", "cs_unicycle")

# Fixed number of Ruiz equilibration iterations to apply before solving a problem
EQUILIBRATION_ITERS = 10


def equilibrate_problem(
    P: Float[Tensor, "n n"],
    q: Float[Tensor, " n"],
    A: Float[Tensor, "m n"],
    b: Float[Tensor, " m"],
    cones: list[Cone],
    max_iter: int = 10,
):
    """Apply fixed-iteration Ruiz equilibration to the problem data in place."""
    if P.is_sparse and not P.is_coalesced():
        P = P.coalesce()
    if A.is_sparse and not A.is_coalesced():
        A = A.coalesce()
    equilibriate_(P, q, A, b, cones, max_iter=max_iter, conv_tol=0.0)
    return P, q, A, b


def nnz(T: torch.Tensor) -> int:
    return T._nnz() if T.is_sparse or T.is_sparse_csr else T.numel()


def print_running_stats(saved: int, total: int, iters_hist: list[int]) -> None:
    """Print running solve statistics after a problem is saved."""
    msg = f"  [saved {saved}/{total}]"
    if iters_hist:
        arr = np.asarray(iters_hist, dtype=float)
        msg += (
            f" iters to converge: last={int(arr[-1])}, avg={arr.mean():.1f}, min={int(arr.min())}, max={int(arr.max())}"
        )
    print(msg)


def sample_kwargs(config: DatasetConfig) -> dict:
    """The generator keyword arguments of one instance: the fixed ones plus a draw of the sampled ones."""
    problem_kwargs = dict(config.fixed_kwargs)
    for k, v in config.sampling.items():
        if isinstance(v, list) and len(v) == 2:
            lo, hi = v
            if isinstance(lo, int) and isinstance(hi, int):
                problem_kwargs[k] = np.random.randint(lo, hi + 1)
            else:
                problem_kwargs[k] = np.random.uniform(lo, hi)
        else:
            problem_kwargs[k] = v
    return problem_kwargs


def dual_too_large(y: torch.Tensor, config: DatasetConfig) -> bool:
    """Whether the reference dual exceeds `config.max_dual_norm` (always `False` when unset)."""
    if config.max_dual_norm is None:
        return False
    y_norm = y.norm().item()
    if y_norm > config.max_dual_norm:
        print(f"  Rejected: dual norm {y_norm:.3e} exceeds max_dual_norm {config.max_dual_norm:.3e}.")
        return True
    return False


def save_problem(
    path: pathlib.Path,
    P: Float[Tensor, "n n"],
    q: Float[Tensor, " n"],
    A: Float[Tensor, "m n"],
    b: Float[Tensor, " m"],
    cones: list[Cone],
    result: Solver_Result,
) -> None:
    with path.open("wb") as f:
        pickle.dump((P, q, A, b, cones, result), f)


def run_generic(config: DatasetConfig, save_path: pathlib.Path) -> None:
    if config.solver not in ("cosmo", "clarabel"):
        raise ValueError(
            f"Unknown solver '{config.solver}' for {config.problem_generator}; expected 'cosmo' or 'clarabel'."
        )

    problem_generator = lookup(generators(), config.problem_generator, "problem generator")
    policy = None
    if config.solver == "cosmo":
        policy_class = lookup(policies(), config.policy, "policy")
        policy = policy_class(**config.policy_kwargs)

    iters_hist: list[int] = []
    attempts = 0
    rejections_solve = 0
    rejections_dual = 0
    pbar = tqdm(total=config.n_problems, dynamic_ncols=True)
    current_prob = 1
    while current_prob <= config.n_problems:
        attempts += 1
        problem_kwargs = sample_kwargs(config)

        print(f"\nGenerating problem {current_prob} with kwargs: {problem_kwargs}")
        P, q, A, b, cones = problem_generator(**problem_kwargs)
        m, n = A.shape
        print(f"  Problem dimensions: n={n}, m={m}, N={nnz(P) + nnz(A)}")

        if config.equilibrate:
            P, q, A, b = equilibrate_problem(P, q, A, b, cones, max_iter=EQUILIBRATION_ITERS)
            print(f"  Equilibrated problem data ({EQUILIBRATION_ITERS} iterations).")

        if config.solver == "cosmo":
            solver = Solver(P, q, A, b, prox=ProjectionProx(cones), policy=policy)
            result = solver.solve(**config.solver_kwargs)
            if not result.converged:
                final_res = torch.max(result.dual_res_norm_hist[-1], result.prim_res_norm_hist[-1])
                print(f"  Problem not solved. Final residual norm: {final_res:.2e}")
                rejections_solve += 1
                continue
            print(f"  Problem solved in {result.num_iters} iterations.")
            if dual_too_large(result.y, config):
                rejections_dual += 1
                continue
            iters_hist.append(result.num_iters)
        else:
            try:
                x_opt, y_opt, obj_val, status, stats = solve_with_stats(P, q, A, b, cones, **config.solver_kwargs)
            except Exception as e:
                # Clarabel can fail hard enough to raise rather than return a non-optimal status;
                # treat it like any other failed solve and resample a fresh problem.
                print(f"  Problem not solved. Clarabel raised {type(e).__name__}: {e}")
                rejections_solve += 1
                continue
            if x_opt is None:
                print(f"  Problem not solved. Clarabel status: {status}")
                rejections_solve += 1
                continue
            print(f"  Solved successfully. Objective value: {obj_val:.6f}")
            if dual_too_large(y_opt, config):
                rejections_dual += 1
                continue
            result = make_solver_result(x_opt, y_opt, A, b, obj_val)
            if stats.get("num_iters") is not None:
                iters_hist.append(stats["num_iters"])

        save_problem(save_path / f"problem_{current_prob:05d}.pkl", P, q, A, b, cones, result)
        print_running_stats(current_prob, config.n_problems, iters_hist)
        current_prob += 1
        pbar.update(1)

    pbar.close()
    print("\n" + "=" * 60)
    print("Dataset Generation Complete")
    print("=" * 60)
    print(f"Total attempts:                 {attempts}")
    print(f"Saved problems:                 {config.n_problems}")
    print(f"Rejections (solve failed):      {rejections_solve}")
    print(f"Rejections (dual norm):         {rejections_dual}")
    print(f"Dataset saved to:               {save_path}")
    print("=" * 60)


def run_covariance_steering_scp(config: DatasetConfig, save_path: pathlib.Path) -> None:
    from ..problems.covariance_steering import (
        SCP_Config,
        sample_double_integrator_instance,
        sample_unicycle_instance,
        solve_scp,
    )

    if config.max_dual_norm is not None:
        raise ValueError("max_dual_norm is not supported by the covariance-steering SCP pipeline.")
    if config.sampling:
        raise ValueError("The covariance-steering SCP pipeline takes fixed_kwargs only, not sampling.")

    fixed = dict(config.fixed_kwargs)
    policy = None
    if config.solver == "cosmo":
        policy_class = lookup(policies(), config.policy, "policy")
        policy = policy_class(**config.policy_kwargs)

    # SCP-specific settings
    scp_cfg = SCP_Config(
        max_iters=int(fixed.pop("scp_max_iters", 15)),
        trust_x0=float(fixed.pop("scp_trust_x0", 5.0)),
        trust_u0=float(fixed.pop("scp_trust_u0", 5.0)),
        trust_x_max=float(fixed.pop("scp_trust_x_max", 50.0)),
        trust_u_max=float(fixed.pop("scp_trust_u_max", 50.0)),
        expand=float(fixed.pop("scp_expand", 2.0)),
        shrink=float(fixed.pop("scp_shrink", 0.5)),
        kkt_tol=float(fixed.pop("scp_kkt_tol", 1e-3)),
        solver=config.solver,
        solver_kwargs=config.solver_kwargs,
        policy=policy,
        equilibrate=config.equilibrate,
    )
    if config.problem_generator == "cs_double_integrator":
        sample_instance = sample_double_integrator_instance
    else:
        sample_instance = sample_unicycle_instance

    saved = 0
    total_attempts = 0
    rejections_scp = 0
    total_subproblems = 0
    subproblem_counts: list[int] = []
    pbar = tqdm(total=config.n_problems, dynamic_ncols=True)

    while saved < config.n_problems:
        total_attempts += 1
        dyn, instance = sample_instance(**fixed)

        print(f"\nGenerating covariance-steering instance {saved + 1} [attempt {total_attempts}]")
        try:
            iterates = list(solve_scp(dyn, instance, scp_cfg))
        except Exception as e:
            rejections_scp += 1
            print(f"  Rejected: {e}")
            continue

        for j, it in enumerate(iterates):
            save_problem(
                save_path / f"problem_{saved + 1:05d}_iter_{j:02d}.pkl", it.P, it.q, it.A, it.b, it.cones, it.result
            )
        total_subproblems += len(iterates)
        converged = iterates[-1].converged if iterates else False
        print(f"  SCP yielded {len(iterates)} subproblems (converged={converged}).")
        saved += 1
        subproblem_counts.append(len(iterates))
        print_running_stats(saved, config.n_problems, subproblem_counts)
        pbar.update(1)

    pbar.close()
    print("\n" + "=" * 60)
    print("Dataset Generation Complete")
    print("=" * 60)
    print(f"Total attempts:                 {total_attempts}")
    print(f"Saved instances:                {saved}")
    print(f"Total subproblems saved:        {total_subproblems}")
    print(f"Rejections (SCP failure):       {rejections_scp}")
    print(f"Success rate:                   {saved / total_attempts * 100:.1f}%")
    print(f"Dataset saved to:               {save_path}")
    print("=" * 60)


def write_dataset_record(config: DatasetConfig, save_path: pathlib.Path) -> pathlib.Path:
    """Write `dataset.json`: the generating config and the number of problem files."""
    record = {
        "generator_config": config.model_dump(mode="json"),
        "n_files": len(list(save_path.glob("problem_*.pkl"))),
    }
    path = save_path / "dataset.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dco.cli.generate", description="Generate a dataset of cone programs."
    )
    parser.add_argument("config_path", type=str, help="Path to the dataset config YAML.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Generate a cone program dataset from a config file."""
    args = build_parser().parse_args(argv)
    config = load_dataset_config(args.config_path)

    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    torch.set_default_device(torch.device(config.device))
    torch.set_default_dtype(torch.float64 if config.precision == "double" else torch.float32)

    save_path = resolve_path(config.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if config.problem_generator in SCP_GENERATORS:
        run_covariance_steering_scp(config, save_path)
    else:
        run_generic(config, save_path)

    print(f"Dataset record:                 {write_dataset_record(config, save_path)}")


if __name__ == "__main__":
    main()
