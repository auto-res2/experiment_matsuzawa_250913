import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import yaml

import src.train as _train  # absolute import avoids relative-import issues

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # project root
CFG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / ".research" / "iteration1"


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

def run_smoke():
    cfg = load_yaml(CFG_DIR / "smoke_test.yaml")
    print("Running SAFE-Swarm 2.0 – SMOKE TEST mode")

    _train.run_exp1(cfg, RESULTS_DIR)
    # Smoke test only runs Exp-1


def run_full():
    cfg = load_yaml(CFG_DIR / "full_experiment.yaml")
    print("Running SAFE-Swarm 2.0 – FULL EXPERIMENT mode")

    # Phase-1: quick smoke to ensure pipeline works
    smoke_ok = True
    try:
        _train.run_exp1(cfg, RESULTS_DIR)
    except Exception as e:  # noqa: BLE001 (broad ok – want to stop early)
        smoke_ok = False
        print(f"[ERROR] Smoke phase failed – aborting full experiment. Reason: {e}")

    if not smoke_ok:
        sys.exit(1)

    # Phase-2: full run (currently only exp1 implemented)
    _train.run_exp1(cfg, RESULTS_DIR)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAFE-Swarm 2.0 Experiment Runner")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="run quick validation")
    g.add_argument("--full-experiment", action="store_true", help="run full experiment")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke()
    else:
        run_full()


if __name__ == "__main__":
    main()
