import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # noqa: F401  (needed for seaborn-matplotlib style)
import torch  # noqa: F401  (kept for future tensor-based metrics)

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
    """Simple helper that queries *controller.evaluate* for each task loader."""
    res = {"task_accuracies": [], "forgetting_metrics": []}
    for tid, loader in enumerate(test_loaders):
        acc = controller.evaluate(loader)
        res["task_accuracies"].append(acc)
        if tid:
            res["forgetting_metrics"].append(max(res["task_accuracies"][:-1]) - acc)
    return res


# -----------------------------------------------------------------------------
# Visualisation helpers  (paths updated to iteration9 as per spec) --------------
_IMAGES_ROOT = Path(".research/iteration9/images")


def _ensure_dir(p: Path):
    """Create directory *p* if it does not already exist."""
    p.mkdir(parents=True, exist_ok=True)


def visualize_results(results: Dict, save_dir: str | Path = _IMAGES_ROOT):
    """Plot task accuracies and save the figure under `.research/iteration9/images/`.

    Parameters
    ----------
    results : Dict
        Must contain key `task_accuracies` that maps to a list of floats 8 [0,1].
    save_dir : str | Path, optional
        Directory in which to save the figure.  Defaults to `_IMAGES_ROOT` so that
        all experiments end up in the mandated location.
    """
    out = Path(save_dir)
    _ensure_dir(out)
    plt.style.use("seaborn-v0_8-paper")

    if "task_accuracies" in results and results["task_accuracies"]:
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
        print(f" saved figure {fname}")


# -----------------------------------------------------------------------------
# Tables 6 JSON utils
# -----------------------------------------------------------------------------

def generate_comparison_table(methods: Dict[str, Dict]):
    """Return a pretty *pandas* table comparing several methods."""
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
    """Recursively convert *obj* so that *json.dump* accepts it."""
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
    """Save *obj* to *path* in JSON format (parent directories auto-created)."""
    p = Path(path)
    _ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2)
    print(f" saved {p}")


# -----------------------------------------------------------------------------
# Make discoverable under multiple import paths (`import evaluate` and
# `import src.evaluate` both succeed without copying files).
# -----------------------------------------------------------------------------
import sys as _sys
_sys.modules.setdefault("evaluate", _sys.modules[__name__])
_sys.modules.setdefault("src.evaluate", _sys.modules[__name__])