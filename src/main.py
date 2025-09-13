# src/main.py
"""Command-line entry for ORCHID-D⁴ experiments."""

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("orchid-main")

CONFIG_DIR = Path("config")
RESULTS_DIR = Path(".research/iteration9"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_cfg(p: Path) -> Dict[str, Any]:
    if not p.exists():
        logger.error(f"Config not found: {p}"); sys.exit(1)
    return yaml.safe_load(open(p))


def run(cfg: Dict[str, Any], tag: str):
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing datasets …")
    _ = prepare_datasets(cfg)  # placeholder, not used further in stub simulation

    logger.info("Running simulator …")
    _, metrics = train_orchid(cfg, out_dir)

    res_path = out_dir / "orchid_d4_results.json"
    eval_res = evaluate(res_path) if res_path.exists() else {}

    final = {
        "cfg": cfg,
        "metrics": metrics,
        "eval": eval_res,
        "tag": tag,
        "ts": time.time(),
    }
    fp = RESULTS_DIR / f"final_{tag}_{int(time.time())}.json"
    json.dump(final, open(fp, "w"), indent=2)
    print(f"Results saved → {fp}")


# ------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Run ORCHID-D⁴ experiments")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="run smoke test config")
    g.add_argument("--full-experiment", action="store_true", help="run full experiment config")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg_file = "smoke_test.yaml" if args.smoke_test else "full_experiment.yaml"
    tag = "smoke" if args.smoke_test else "full"
    cfg = load_cfg(CONFIG_DIR / cfg_file)
    if args.seed != 42:
        cfg["seed"] = args.seed
    try:
        run(cfg, tag)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failure: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
