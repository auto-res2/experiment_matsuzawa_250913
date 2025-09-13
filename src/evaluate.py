import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

__all__ = [
    "evaluate_continual_learning",
    "visualize_results",
    "generate_comparison_table",
    "save_results_json",
]

# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def evaluate_continual_learning(controller, test_loaders, cfg):
    res = {"task_accuracies": [], "forgetting_metrics": []}
    for tid, loader in enumerate(test_loaders):
        acc = controller.evaluate(loader)
        res["task_accuracies"].append(acc)
        if tid:
            res["forgetting_metrics"].append(
                max(res["task_accuracies"][:-1]) - acc
            )
    return res


# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------

_IMAGES_ROOT = Path(".research/iteration2/images")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def visualize_results(results: Dict, save_dir: str | Path = _IMAGES_ROOT):
    """Plot task accuracies (and other metrics in the future) and save under
    .research/iteration2/images/…
    """
    out = Path(save_dir)
    _ensure_dir(out)
    plt.style.use("seaborn-v0_8-paper")

    if "task_accuracies" in results:
        t = np.arange(1, len(results["task_accuracies"]) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(t, results["task_accuracies"], marker="o")
        plt.xlabel("Task")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fname = out / "task_acc.pdf"
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"✓ saved figure {fname}")


# -----------------------------------------------------------------------------
# Tables & JSON utils
# -----------------------------------------------------------------------------

def generate_comparison_table(methods: Dict[str, Dict]):
    rows = []
    for name, stats in methods.items():
        rows.append(
            {
                "Method": name,
                "Avg Acc": stats.get("avg_accuracy", 0) * 100,
                "Worst Acc": stats.get("worst_accuracy", 0) * 100,
                "Energy/Acc (mJ)": stats.get("avg_energy_per_correct", 0),
                "SRAM Overshoots": stats.get("total_sram_overshoots", 0),
            }
        )
    df = pd.DataFrame(rows).round(2).sort_values("Avg Acc", ascending=False)
    return df


# ----- JSON serialisation -----------------------------------------------------

def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(i) for i in obj]
    return obj


def save_results_json(obj: Dict, path: str | Path):
    p = Path(path)
    _ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2)
    print(f"✓ saved {p}")
