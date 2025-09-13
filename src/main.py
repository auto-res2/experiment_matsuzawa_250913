import argparse
import importlib.util
import importlib.machinery
from pathlib import Path
import sys
import yaml

# -----------------------------------------------------------------------------
_pkg_root = Path(__file__).resolve().parent  # .../src
_repo_root = _pkg_root.parent               # project root

if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# -----------------------------------------------------------------------------
# Dynamic module loader able to handle missing *.py* suffix.
# -----------------------------------------------------------------------------

def _load_from_path(mod_name: str, path: Path):
    loader = importlib.machinery.SourceFileLoader(mod_name, str(path))
    spec = importlib.util.spec_from_loader(mod_name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    sys.modules[mod_name] = module
    return module

# -----------------------------------------------------------------------------
# Try standard imports; fall back to manual loading otherwise.
# -----------------------------------------------------------------------------
try:
    from train_py import train_experiment, create_resource_shock_trace  # type: ignore
    from preprocess_py import DataPreprocessor  # type: ignore
    from evaluate_py import (
        generate_comparison_table,
        visualize_results,
        save_results_json,
    )  # type: ignore
except ImportError:
    # train_py
    for _cand in [
        _pkg_root / "train_py.py",
        _pkg_root / "train_py",
        _repo_root / "train_py.py",
        _repo_root / "train_py",
    ]:
        if _cand.exists():
            _mod = _load_from_path("train_py", _cand)
            train_experiment = _mod.train_experiment  # type: ignore
            create_resource_shock_trace = _mod.create_resource_shock_trace  # type: ignore
            break
    else:
        raise ImportError("Could not locate train_py module.")

    # evaluate_py
    for _cand in [
        _pkg_root / "evaluate_py.py",
        _pkg_root / "evaluate_py",
        _repo_root / "evaluate_py.py",
        _repo_root / "evaluate_py",
    ]:
        if _cand.exists():
            _mod = _load_from_path("evaluate_py", _cand)
            generate_comparison_table = _mod.generate_comparison_table  # type: ignore
            visualize_results = _mod.visualize_results  # type: ignore
            save_results_json = _mod.save_results_json  # type: ignore
            break
    else:
        raise ImportError("Could not locate evaluate_py module.")

    # preprocess_py
    for _cand in [
        _pkg_root / "preprocess_py.py",
        _pkg_root / "preprocess_py",
        _repo_root / "preprocess_py.py",
        _repo_root / "preprocess_py",
    ]:
        if _cand.exists():
            _mod = _load_from_path("preprocess_py", _cand)
            DataPreprocessor = _mod.DataPreprocessor  # type: ignore
            break
    else:
        raise ImportError("Could not locate preprocess_py module.")

# -----------------------------------------------------------------------------
# Paths – mandated iteration11 folders
# -----------------------------------------------------------------------------
CFG_DIR = _repo_root / "config"
JSON_ROOT = Path(".research/iteration11")
IMG_ROOT = JSON_ROOT / "images"


def _load_cfg(name: str):
    for p in [CFG_DIR / name, _repo_root / name]:
        if p.exists():
            with open(p, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Configuration file '{name}' not found.")


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
    parser = argparse.ArgumentParser(description="FoReCoast-CL experiment runner")
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
