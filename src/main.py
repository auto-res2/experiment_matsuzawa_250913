"""src/main.py
Entry-point with CLI that orchestrates smoke/full experiments.

Usage:
    uv run python -m src.main --smoke-test
    uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import pathlib
import yaml
from typing import Dict, Any

from .evaluate import EXPERIMENT_REGISTRY

################################################################################
#   Config loader (local helper → no extra file needed)
################################################################################

def _load_yaml(path: str | pathlib.Path) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file {p} not found.")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

################################################################################
#   CLI
################################################################################

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HAWQ-RAPTOR experimental runner")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smoke-test", action="store_true", help="run quick smoke test")
    grp.add_argument("--full-experiment", action="store_true", help="run full experiment")
    return parser.parse_args(argv)


def _run():
    args = _parse_args()
    cfg_path = pathlib.Path("config/smoke_test.yaml" if args.smoke_test else "config/full_experiment.yaml")
    cfg = _load_yaml(cfg_path)

    # iterate in declared order – smoke test first, then possibly full (requirement)
    for exp_name in cfg["run_order"]:
        if exp_name not in EXPERIMENT_REGISTRY:
            raise RuntimeError(f"Unknown experiment '{exp_name}'.")
        print(f"Running {exp_name} …", flush=True)
        EXPERIMENT_REGISTRY[exp_name](cfg[exp_name])


if __name__ == "__main__":  # pragma: no cover
    _run()
