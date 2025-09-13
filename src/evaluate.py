import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

__all__ = [
    "compute_statistics",
    "perform_significance_test",
    "visualize_results",
    "generate_comparison_table",
    "save_results_json",
]


def compute_statistics(arr: List[float], conf: float = 0.95) -> Dict:
    if not arr:
        return {"mean": 0, "std": 0, "ci_lower": 0, "ci_upper": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) > 1:
        lo, hi = stats.t.interval(conf, len(arr) - 1, loc=mean, scale=std / np.sqrt(len(arr)))
    else:
        lo = hi = mean
    return {"mean": mean, "std": std, "ci_lower": float(lo), "ci_upper": float(hi)}


def perform_significance_test(a: List[float], b: List[float]) -> Dict:
    if len(a) < 2 or len(b) < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}
    t, p = stats.ttest_rel(a, b)
    return {"t_statistic": float(t), "p_value": float(p), "significant": p < 0.05}


def visualize_results(res: Dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-paper")
    if "task_accuracies" in res and res["task_accuracies"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        t = np.arange(1, len(res["task_accuracies"]) + 1)
        ax.plot(t, np.array(res["task_accuracies"]) * 100, "o-", lw=2)
        ax.set_xlabel("Task")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(save_dir / "task_accuracy.pdf", dpi=300)
        plt.close(fig)


def generate_comparison_table(methods: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for name, s in methods.items():
        rows.append(
            {
                "Method": name,
                "Avg Acc (%)": f"{s.get('avg_accuracy', 0)*100:.1f}",
                "Worst Acc (%)": f"{s.get('worst_accuracy', 0)*100:.1f}",
                "Energy/Acc (mJ)": f"{s.get('avg_energy_per_correct', 0):.3f}",
                "SRAM Overshoots": int(s.get("total_sram_overshoots", 0)),
            }
        )
    return pd.DataFrame(rows)


def save_results_json(obj: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        if isinstance(o, dict):
            return {k: _conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_conv(v) for v in o]
        return o

    with open(path, "w") as f:
        json.dump(_conv(obj), f, indent=2)
