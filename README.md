# Deep Conic Optimization

Official code release for [**Scalable Deep Unfolding of Conic Optimizers**](https://arxiv.org/abs/2606.13825), presented at the 2026 Conference on Robot Learning.

## Abstract

Deep unfolding (DU) accelerates iterative optimizers by training learnable components through unrolled iterations, but extending DU to the large-scale semidefinite programs (SDPs) common in robotics has remained limited. Unrolling a full-update conic solver such as COSMO exposes two obstacles: backpropagating through the per-iteration linear-system solve incurs memory quadratic in the problem size when the coefficient matrix is formed explicitly, and backpropagating through the positive semidefinite (PSD) cone projection is numerically unstable under repeated eigenvalues. We address the first with a matrix-free implicit differentiation rule that reduces memory from $O(n^2)$ to $O(n + \mathrm{nnz}(A))$, and the second with a stable backward rule based on the Dalečkii-Krein representation of the Fréchet derivative. Together, these enable learning lightweight hyperparameter policies and warm starts for COSMO. On standalone SDPs and second-order cone programs, ranging from max-cut and Lovász $\vartheta$ to robust estimation and control, the learned policies outperform tuned COSMO on every problem class, reducing iterations to tolerance by up to 60×, and transfer without retraining to instances up to 10× larger in the size parameter while retaining 2-5× speedups. Our matrix-free solver implementation also batches naturally on the GPU, solving 64 Lovász $\vartheta$ SDPs 14× faster than sequential COSMO.jl. When embedded in sequential convex programming for nonlinear covariance steering, our learned optimizers outperform the interior-point solver Clarabel in terms of both success rate and solve time.

## Installation

This package requires `python>=3.14` and a CUDA-compatible GPU. PyTorch with CUDA will be installed automatically from PyPI. We recommend using a package manager such as `conda` or `uv`. This guide will use `conda`.

1. Start by cloning the repo and setting up your python environment:
    ```sh
    git clone git@github.com:alexoshin/deep-conic-optimization.git
    cd deep-conic-optimization

    conda create -n dco_env python=3.14 -y
    conda activate dco_env
    pip install -e ".[dev]"  # drop [dev] if you don't need pytest/ruff
    ```

2. Verify the installation by running the following commands:
    ```sh
    python -c "import dco, torch; print(torch.cuda.is_available())"
    dco-train --help
    pytest -q
    ```
    This checks that the python module is importable, the command-line scripts are callable, and the unit tests all pass successfully.

## Usage

We provide the configs for reproducing the datasets and models from the paper (see `configs/`), along with a few command line helpers for generation, training, and evaluation.

### Generate datasets
```sh
dco-generate configs/datasets/train/max_cut_n50.yaml      # 100 training instances, 20 validation instances
dco-generate configs/datasets/test/max_cut_n50_test.yaml  # 100 test instances
```

### Train learned policies
```sh
dco-train configs/training/max_cut_n50.yaml  # Results saved to outputs/max_cut_n50/<timestamp>/
```

### Benchmark learned policies
```sh
dco-benchmark outputs/max_cut_n50/<timestamp> --dataset datasets/max_cut_n50/test100
```

## Citation

If you found this project helpful for your work, please consider citing our paper:

```
@inproceedings{Oshin2026Scalable,
    title     = {Scalable Deep Unfolding of Conic Optimizers},
    author    = {Oshin, Alex and Ghosh, Rahul Vodeb and Theodorou, Evangelos A.},
    booktitle = {Proceedings of the 10th Conference on Robot Learning},
    year      = {2026},
    publisher = {PMLR},
    url       = {https://openreview.net/forum?id=CSYA6oHdlE}
}
```
