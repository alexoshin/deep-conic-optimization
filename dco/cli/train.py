"""Train a learned COSMO model.

Usage:
    dco-train <config_path>
"""

import argparse

from ..train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned hyperparameter policy and warm start.")
    parser.add_argument("config_path", help="training config YAML")
    args = parser.parse_args()
    train(args.config_path)


if __name__ == "__main__":
    main()
