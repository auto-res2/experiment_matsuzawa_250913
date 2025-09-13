"""Command-line runner orchestrating smoke / full experiment."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from .evaluate import evaluate
from .preprocess import prepare_datasets
from .train import train_orchid

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_DIR = Path("config")
RESULTS_DIR = Path(".research/iteration7")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# util
# -----------------------------------------------------------------------------

def _load_cfg(p: Path) -> Dict[str, Any]:
    if not p.exists():
        logger.error("Config not found: %s", p); sys.exit(1)
    return yaml.safe_load(p.read_text())


# -----------------------------------------------------------------------------
# main experiment routine
# -----------------------------------------------------------------------------

def _run(cfg: Dict[str, Any], tag: str):
    torch.manual_seed(42); np.random.seed(42)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # prepare (unused heavy datasets skipped in smoke)
    prepare_datasets(cfg)

    # simulate / train
    model, metrics = train_orchid(cfg, out_dir)

    # evaluation (produces images)
    res_path = out_dir / "orchid_d4_results.json"
    ev = evaluate(res_path) if res_path.exists() else {}

    # aggregate & persist
    payload = {
        "config": cfg,
        "metrics": metrics,
        "evaluation": ev,
        "timestamp": time.time(),
    }
    save = RESULTS_DIR / f"results_{tag}_{int(time.time())}.json"
    save.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():  # noqa: D401
    ap = argparse.ArgumentParser("ORCHID-D⁴ runner")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true")
    g.add_argument("--full-experiment", action="store_true")
    args = ap.parse_args()

    cfg_path = CONFIG_DIR / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    tag = "smoke" if args.smoke_test else "full"
    cfg = _load_cfg(cfg_path)

    try:
        _run(cfg, tag)
    except Exception as exc:  # pragma: no cover
        logger.error("Experiment failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
