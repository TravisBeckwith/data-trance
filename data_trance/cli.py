"""
data-trance CLI
================
    data-trance run config.yaml                 -> all configured steps in order
    data-trance step assess config.yaml          -> just that one step
"""

import argparse
import sys

from .config import VALID_STEPS, load_config
from .context import Context
from .steps import apply_transform, assess, detect_type, recommend, report, validate

STEP_MODULES = {
    "detect_type": detect_type,
    "assess": assess,
    "recommend": recommend,
    "apply_transform": apply_transform,
    "validate": validate,
    "report": report,
}


def run_pipeline(config_path: str, only_step: str = None):
    config = load_config(config_path)
    ctx = Context(config)

    steps_to_run = [only_step] if only_step else config.steps
    for step_name in steps_to_run:
        if step_name not in STEP_MODULES:
            sys.exit(f"Unknown step '{step_name}'. Valid steps: {VALID_STEPS}")
        print(f"-> running step: {step_name}")
        try:
            STEP_MODULES[step_name].run(ctx)
        except FileNotFoundError as e:
            sys.exit(f"error: {e}")
    print(f"\nDone. Output written to {config.output_dir}/")


def main():
    parser = argparse.ArgumentParser(prog="data-trance",
                                      description="Assess a distribution, then transform it.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full pipeline as configured")
    p_run.add_argument("config", help="Path to the YAML config file")

    p_step = sub.add_parser("step", help="Run a single step")
    p_step.add_argument("name", choices=VALID_STEPS)
    p_step.add_argument("config", help="Path to the YAML config file")

    p_train = sub.add_parser("train-recommender",
                              help="(Re)train the ML/DL transform recommenders "
                                   "on synthetic data")
    p_train.add_argument("--n-samples", type=int, default=4000,
                          help="Number of synthetic training samples (default 4000)")
    p_train.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args.config)
    elif args.command == "step":
        run_pipeline(args.config, only_step=args.name)
    elif args.command == "train-recommender":
        from .ml.train import main as train_main
        train_main(n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
