import argparse
from pathlib import Path
import sys
import yaml

# -----------------------------------------------------------------------------
# Make intra-package imports robust whether main.py is executed via
# `python -m src.main` *or* as a script path `python src/main.py`.
# -----------------------------------------------------------------------------
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

try:
    # When run as a module (python -m src.main) – regular relative imports work
    from .train import train_experiment, create_resource_shock_trace  # type: ignore
    from .preprocess_py import DataPreprocessor  # type: ignore
    from .evaluate import (
        generate_comparison_table,
        visualize_results,
        save_results_json,
    )
except ImportError:
    # Fallback when executed as a script (python src/main.py)
    from train import train_experiment, create_resource_shock_trace  # type: ignore
    from preprocess_py import DataPreprocessor  # type: ignore
    from evaluate import (
        generate_comparison_table,
        visualize_results,
        save_results_json,
    )

# -----------------------------------------------------------------------------
CFG_DIR = Path(__file__).resolve().parent.parent / "config"
JSON_ROOT = Path(".research/iteration2")
IMG_ROOT = Path(".research/iteration2/images")


def _load_cfg(name: str):
    with open(CFG_DIR / name, "r") as f:
        return yaml.safe_load(f)


def _prepare_dirs():
    for d in [JSON_ROOT, IMG_ROOT, Path("checkpoints")]:
        d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Experiment wrappers
# -----------------------------------------------------------------------------

def smoke_test():
    cfg = _load_cfg("smoke_test.yaml")
    pre = DataPreprocessor("vision")
    shock = create_resource_shock_trace(3000)

    ctrl, res = train_experiment(cfg["forecoast"], pre, shock)
    print("Smoke-test AvgAcc:", res["final_stats"]["avg_accuracy"])

    visualize_results(res["final_stats"], IMG_ROOT / "smoke")
    json_path = JSON_ROOT / "smoke_results.json"
    save_results_json(res, json_path)
    print(json_path.read_text())


def full_experiment():
    cfg = _load_cfg("full_experiment.yaml")
    pre = DataPreprocessor("vision")
    shock = create_resource_shock_trace()

    ctrl, res = train_experiment(cfg["forecoast"], pre, shock)
    tbl = generate_comparison_table({"FoReCoast-CL": res["final_stats"]})
    print(tbl.to_string(index=False))

    visualize_results(res["final_stats"], IMG_ROOT / "full")
    json_path = JSON_ROOT / "full_results.json"
    save_results_json(res, json_path)
    print(json_path.read_text())


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

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
