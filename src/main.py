import argparse
import importlib.util
from pathlib import Path
import sys
import yaml

# -----------------------------------------------------------------------------
# Make intra-package imports robust whether main.py is executed via
# `python -m src.main` *or* as a script path `python src/main.py` *or* after a
# pip install where some source files might be missing from site-packages.
# -----------------------------------------------------------------------------
_pkg_root = Path(__file__).resolve().parent  # .../src
_repo_root = _pkg_root.parent               # project root

# Ensure the *repository* root is on sys.path so that we can always fall back
# to loading loose source files even if they were not included in the wheel.
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Try the standard relative imports first --------------------------------------
try:
    from .train import train_experiment, create_resource_shock_trace  # type: ignore
    from .preprocess_py import DataPreprocessor  # type: ignore
    from .evaluate import (
        generate_comparison_table,
        visualize_results,
        save_results_json,
    )
except ImportError:
    # -------------------------------------------------------------------------
    # Fallback strategy: import from loose files sitting at the repo root.
    # -------------------------------------------------------------------------
    # ---- train ----------------------------------------------------------------
    try:
        from train import train_experiment, create_resource_shock_trace  # type: ignore
    except ImportError:
        _train_candidates = [_pkg_root / "train_py.py", _repo_root / "train_py.py"]
        for _cand in _train_candidates:
            if _cand.exists():
                spec = importlib.util.spec_from_file_location("train", _cand)
                assert spec and spec.loader, f"Could not create spec for {_cand}"
                _mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_mod)  # type: ignore[arg-type]
                sys.modules["train"] = _mod
                train_experiment = _mod.train_experiment  # type: ignore
                create_resource_shock_trace = _mod.create_resource_shock_trace  # type: ignore
                break
        else:
            raise
    # ---- evaluate -------------------------------------------------------------
    try:
        from evaluate import generate_comparison_table, visualize_results, save_results_json  # type: ignore
    except ImportError:
        _eval_candidates = [_pkg_root / "evaluate_py.py", _repo_root / "evaluate_py.py"]
        for _cand in _eval_candidates:
            if _cand.exists():
                spec = importlib.util.spec_from_file_location("evaluate", _cand)
                assert spec and spec.loader, f"Could not create spec for {_cand}"
                _mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_mod)  # type: ignore[arg-type]
                sys.modules["evaluate"] = _mod
                generate_comparison_table = _mod.generate_comparison_table  # type: ignore
                visualize_results = _mod.visualize_results  # type: ignore
                save_results_json = _mod.save_results_json  # type: ignore
                break
        else:
            raise
    # ---- preprocess -----------------------------------------------------------
    try:
        from preprocess_py import DataPreprocessor  # type: ignore
    except ImportError:
        _preprocess_path_candidates = [
            _pkg_root / "preprocess_py.py",
            _repo_root / "preprocess_py.py",
        ]
        for _cand in _preprocess_path_candidates:
            if _cand.exists():
                spec = importlib.util.spec_from_file_location("preprocess_py", _cand)
                assert spec and spec.loader, f"Could not create spec for {_cand}"
                _mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_mod)  # type: ignore[arg-type]
                sys.modules["preprocess_py"] = _mod
                DataPreprocessor = _mod.DataPreprocessor  # type: ignore
                break
        else:
            raise ImportError(
                "Failed to import DataPreprocessor: no preprocess_py module found "
                "in installed package nor as a loose source file."
            )

# -----------------------------------------------------------------------------
# Paths (updated to iteration6 as mandated) ------------------------------------
CFG_DIR = _repo_root / "config"
JSON_ROOT = Path(".research/iteration6")
IMG_ROOT = Path(".research/iteration6/images")


def _load_cfg(name: str):
    """Robust YAML loader."""
    candidates = [CFG_DIR / name, _repo_root / name]
    for p in candidates:
        if p.exists():
            with open(p, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Configuration file '{name}' not found in {candidates}")


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
