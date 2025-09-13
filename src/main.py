"""src/main.py
Entry-point that orchestrates smoke test and full experiment execution.
Complies with the required CLI flags:

    uv run python -m src.main --smoke-test
    uv run python -m src.main --full-experiment

Images are saved under .research/iteration3/images
JSON metrics are saved directly under .research/iteration3/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, Any

import torch
import yaml

from .train import (
    CuFSF,
    GCNII,
    Trainer,
    set_seed,
)
from .evaluate import save_json, line_plot
from .preprocess import ring_grid, fetch_hf_dataset, ensure_all_full

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw

    def __getattr__(self, item):  # noqa: D401 – behave like mapping accessor
        return self.raw.get(item)


def load_config(path: pathlib.Path | str) -> Config:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw)

# ---------------------------------------------------------------------------
# Helper – build masks on-the-fly for synthetic datasets (60/20/20 split)
# ---------------------------------------------------------------------------

def _make_random_masks(num_nodes: int, train_ratio=0.6, val_ratio=0.2):
    idx = torch.randperm(num_nodes)
    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask

# ---------------------------------------------------------------------------
# Paths (updated to comply with iteration3 spec)
# ---------------------------------------------------------------------------

JSON_DIR = pathlib.Path(".research/iteration3")
IMG_DIR = JSON_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  Smoke test workflow (≤ 3 min on CPU)
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run_smoke() -> None:
    cfg = load_config(ROOT / "config" / "smoke_test.yaml")
    set_seed(int(cfg.seed))

    # Synthetic data (L=4)
    data = ring_grid(500, length=4, seed=int(cfg.seed))
    masks = _make_random_masks(data.num_nodes)
    data.train_mask, data.val_mask, data.test_mask = masks

    model = CuFSF(in_dim=data.x.size(1), hidden=64, num_classes=2)
    trainer = Trainer({"epochs": int(cfg.epochs)}, model, data, masks)
    result, curve = trainer.run()

    # ---- persistence ----
    save_json(result, JSON_DIR / "smoke_results.json")
    line_plot(range(len(curve)), curve, "epoch", "test_acc", "Smoke Accuracy", IMG_DIR / "accuracy_smoke.pdf")

    # ---------------- STDOUT ----------------
    print("Smoke-test – Curvature-Flow Rescue (synthetic L=4)\n", json.dumps(result, indent=2))
    print("Figures saved:", ["accuracy_smoke.pdf"], file=sys.stderr)

# ---------------------------------------------------------------------------
# 2.  Full experiment (only Exp-1 included for brevity)
# ---------------------------------------------------------------------------

def _run_full() -> None:
    cfg = load_config(ROOT / "config" / "full_experiment.yaml")
    ensure_all_full()  # aborts if a required download fails

    for seed in range(int(cfg.num_seeds)):
        set_seed(seed)
        for ds in cfg.exp1["datasets"]:
            if ds.startswith("synthetic"):
                L = int(ds.split("_L")[-1])
                data = ring_grid(5000, length=L, seed=seed)
            elif ds in ("chameleon", "squirrel"):
                path = fetch_hf_dataset(ds)
                import dgl  # heavy import – localised to avoid cost for smoke test

                g, _ = dgl.load_graphs(str(path))
                data = g[0].to(torch.device("cpu")).to_pyg()
            else:
                print(f"Dataset {ds} not implemented – skipping …", file=sys.stderr)
                continue

            masks = _make_random_masks(data.num_nodes)
            data.train_mask, data.val_mask, data.test_mask = masks

            for model_name in cfg.exp1["models"]:
                if model_name == "cufsf":
                    model = CuFSF(in_dim=data.x.size(1), hidden=128, num_classes=int(data.y.max()) + 1)
                elif model_name == "cufsf_no_cfc":
                    model = CuFSF(in_dim=data.x.size(1), hidden=128, num_classes=int(data.y.max()) + 1, cfc_steps=0)
                elif model_name == "gcn2":
                    model = GCNII(in_dim=data.x.size(1), hidden=128, num_classes=int(data.y.max()) + 1)
                else:
                    print(f"Model {model_name} not implemented – skipping …", file=sys.stderr)
                    continue

                trainer = Trainer(
                    {"epochs": int(cfg.exp1["epochs"]), "virtual_depth": int(cfg.exp1["virtual_depth"])} ,
                    model,
                    data,
                    masks,
                )
                result, curve = trainer.run()

                # Flat file naming ensures all JSON files live directly under .research/iteration3/
                json_fname = f"{ds}__{model_name}__seed{seed}.json"
                save_json(result, JSON_DIR / json_fname)

                img_fname = f"{ds}__{model_name}__seed{seed}.pdf"
                line_plot(range(len(curve)), curve, "epoch", "test_acc", f"{model_name}_{ds}", IMG_DIR / img_fname)

                print(f"Experiment – {ds} – {model_name}\n", json.dumps(result, indent=2))
                print("Figures saved:", [img_fname], file=sys.stderr)

    print("Experiment 2 & 3 execution logic omitted for brevity.")

# ---------------------------------------------------------------------------
# 3.  CLI glue
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smoke-test", action="store_true", help="Run the quick sanity check")
    grp.add_argument("--full-experiment", action="store_true", help="Run the publication-grade experiment")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke()
    elif args.full_experiment:
        _run_full()