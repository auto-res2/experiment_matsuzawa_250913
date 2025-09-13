import argparse
import os
from pathlib import Path

import yaml

from .train import train_experiment, create_resource_shock_trace
from .preprocess_py import DataPreprocessor  # type: ignore
from .evaluate import generate_comparison_table, visualize_results, save_results_json

CFG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_cfg(name: str):
    with open(CFG_DIR / name, "r") as f:
        return yaml.safe_load(f)


def _prepare_dirs():
    for d in ["results", "figures", "checkpoints", ".research/iteration1/images"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def smoke_test():
    cfg = _load_cfg("smoke_test.yaml")
    pre = DataPreprocessor("vision")
    shock = create_resource_shock_trace(3000)

    ctrl, res = train_experiment(cfg["forecoast"], pre, shock)
    print("Smoke-test AvgAcc:", res["final_stats"]["avg_accuracy"])

    visualize_results(res["final_stats"], "figures/smoke")
    save_results_json(res, "results/smoke_results.json")
    print(Path("results/smoke_results.json").read_text())


def full_experiment():
    cfg = _load_cfg("full_experiment.yaml")
    pre = DataPreprocessor("vision")
    shock = create_resource_shock_trace()

    ctrl, res = train_experiment(cfg["forecoast"], pre, shock)
    tbl = generate_comparison_table({"FoReCoast-CL": res["final_stats"]})
    print(tbl.to_string(index=False))

    visualize_results(res["final_stats"], "figures/full")
    save_results_json(res, "results/full_results.json")
    print(Path("results/full_results.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="run quick smoke-test")
    parser.add_argument("--full-experiment", action="store_true", help="run full experiment")
    args = parser.parse_args()
    _prepare_dirs()

    if args.smoke_test:
        smoke_test()
    elif args.full_experiment:
        full_experiment()
    else:
        print("Specify --smoke-test or --full-experiment")


if __name__ == "__main__":
    main()
