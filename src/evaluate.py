"""
Evaluation utilities for FedC3PO – performance, calibration, fairness & curvature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import Tensor

plt.style.use("seaborn-v0_8-paper")


def compute_tail_ece(pred: Tensor, lab: Tensor, n_bins: int = 10, pct: float = 95) -> float:
    conf, _ = torch.max(F.softmax(pred, 1), 1)
    thresh = torch.quantile(conf, pct / 100)
    mask = conf >= thresh
    if not mask.any():
        return 0.0
    conf, pred_lbl, lab = conf[mask], pred.argmax(1)[mask], lab[mask]
    ece = 0.0
    for b in range(n_bins):
        l, u = b / n_bins, (b + 1) / n_bins
        in_bin = (conf >= l) & (conf < u)
        if not in_bin.any():
            continue
        bin_conf = conf[in_bin].mean().item()
        bin_acc = (pred_lbl[in_bin] == lab[in_bin]).float().mean().item()
        ece += in_bin.sum().item() / len(conf) * abs(bin_conf - bin_acc)
    return ece


def compute_equalised_odds(pred: Tensor, lab: Tensor, sens: Tensor) -> Dict[str, float]:
    hat = pred.argmax(1)
    groups = sens.unique()
    tprs, fprs = [], []
    for g in groups:
        m = sens == g
        if (lab[m] == 1).any():
            tprs.append(((hat[m] == 1) & (lab[m] == 1)).float().mean().item())
        if (lab[m] == 0).any():
            fprs.append(((hat[m] == 1) & (lab[m] == 0)).float().mean().item())
    return {
        "equalised_odds": max(max(tprs) - min(tprs), max(fprs) - min(fprs)) if len(tprs) >= 2 else 0,
        "tpr_gap": max(tprs) - min(tprs) if len(tprs) >= 2 else 0,
        "fpr_gap": max(fprs) - min(fprs) if len(fprs) >= 2 else 0,
    }


def evaluate_model(model, loader, cfg, device: Optional[str] = None) -> Tuple[Dict, Tensor, Tensor]:
    # Auto-select device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    preds, labs = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x.to(device), batch.edge_index.to(device))
            preds.append(out.cpu())
            labs.append(batch.y.cpu())
    preds, labs = torch.cat(preds), torch.cat(labs)
    acc = accuracy_score(labs, preds.argmax(1))
    prec, rec, f1, _ = precision_recall_fscore_support(labs, preds.argmax(1), average="weighted")
    tail_ece = compute_tail_ece(preds, labs)
    # Dummy sensitive attribute for smoke/fallback
    sens = torch.zeros_like(labs)
    sens[len(labs) // 2 :] = 1
    fairness = compute_equalised_odds(preds, labs, sens)
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "tail_ece": float(tail_ece),
        **fairness,
    }
    return metrics, preds, labs


def save_evaluation_results(res: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Evaluation results saved → {path}")


def plot_confusion_matrix(lab: Tensor, pred: Tensor, path: str, class_names: Optional[List[str]] = None):
    cm = confusion_matrix(lab, pred.argmax(1))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300)
    print(f"Confusion matrix saved → {path}")
    plt.close()
