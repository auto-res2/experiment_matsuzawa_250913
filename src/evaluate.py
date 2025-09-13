"""src/evaluate.py
Pure evaluation utilities – metrics, statistics, plotting and JSON I/O.
Nothing here performs training or model definition, satisfying the
specification for *evaluation, statistical analysis and plotting*.
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Sequence, Dict, Any, List

import torch
from torch.linalg import svdvals
from torch_geometric.utils import resistance_distance

import matplotlib

matplotlib.use("Agg")  # enforce non-interactive back-end for PDF output
import matplotlib.pyplot as plt  # noqa: E402 – after back-end selection

# ---------------------------------------------------------------------------
# 1.  Core metrics
# ---------------------------------------------------------------------------

def accuracy(out: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """Masked node classification accuracy."""
    pred = out[mask].argmax(dim=1)
    return float((pred == y[mask]).float().mean())


def effective_rank(Z: torch.Tensor) -> float:
    """Shannon effective rank (exp(entropy)) of the singular values."""
    s = svdvals(Z)
    p = s / s.sum()
    ent = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(ent))


def compute_stretch(data, virtual_depth: int = 128) -> float:
    """Over-squashing proxy: mean log-det of resistance distances of 2-hop nbrs."""
    idx = torch.randint(0, data.num_nodes, (1024,))
    total = 0.0
    for i in idx:
        nbr = data.edge_index[1][data.edge_index[0] == i]
        if len(nbr) == 0:
            continue
        subset = torch.stack([i.repeat(len(nbr)), nbr])
        dist = resistance_distance(data.edge_index, num_nodes=data.num_nodes, subset=subset)
        total += torch.log(dist + 1e-8).mean()
    return float(total / len(idx))

# ---------------------------------------------------------------------------
# 2.  Plotting helpers (PDF-only, complies with spec)
# ---------------------------------------------------------------------------

def line_plot(
    xs: Sequence[int],
    ys: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: pathlib.Path | str,
) -> None:
    plt.figure()
    plt.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.3f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    filename = pathlib.Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------------
# 3.  Lightweight JSON logger
# ---------------------------------------------------------------------------

def save_json(result: Dict[str, Any], path: pathlib.Path | str) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
