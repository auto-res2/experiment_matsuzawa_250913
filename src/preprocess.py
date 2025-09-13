"""src/preprocess.py
Data acquisition and pre-processing utilities live here (synthetic graph
generation, HuggingFace dataset download, etc.).
"""
from __future__ import annotations

import os
import pathlib
from typing import Dict, Tuple

import torch
import random
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from huggingface_hub import hf_hub_download

# Root directory (repository level) ------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  Synthetic ring-grid generator (controls path length L)
# ---------------------------------------------------------------------------

def ring_grid(num_nodes: int = 5000, length: int = 4, seed: int = 0) -> Data:
    random.seed(seed)
    torch.manual_seed(seed)

    # 1-D ring
    edge_index = torch.tensor(
        [list(range(num_nodes)), [(i + 1) % num_nodes for i in range(num_nodes)]],
        dtype=torch.long,
    )
    edge_index = to_undirected(edge_index)

    # Short-cuts every `length` nodes
    shortcut = torch.tensor(
        [list(range(num_nodes)), [(i + length) % num_nodes for i in range(num_nodes)]],
        dtype=torch.long,
    )
    edge_index = torch.cat([edge_index, shortcut], dim=1)
    edge_index = to_undirected(edge_index)

    x = torch.randn(num_nodes, 64) + 0.01 * torch.arange(num_nodes).view(-1, 1).float()
    y = (torch.arange(num_nodes) % 2).long()
    return Data(x=x, edge_index=edge_index, y=y)

# ---------------------------------------------------------------------------
# 2.  HuggingFace datasets used in *real* experiments
# ---------------------------------------------------------------------------

HF_DATASETS: Dict[str, Tuple[str, str]] = {
    "chameleon": ("SauravMaheshkar/pareto-chameleon", "processed/chameleon.bin"),
    "squirrel": ("SauravMaheshkar/pareto-squirrel", "processed/squirrel.bin"),
    "reddit_threads": (
        "graphs-datasets/reddit_threads",
        "data/full-00000-of-00001-f589c8aeb94d15de.parquet",
    ),
}


def _local_dataset_path(filename: str | pathlib.Path) -> pathlib.Path:
    return DATA_DIR / filename


def fetch_hf_dataset(name: str) -> pathlib.Path:
    """Download a dataset from the HuggingFace Hub or return cached path."""
    if name not in HF_DATASETS:
        raise ValueError(f"Unknown dataset key: {name}")
    repo, filename = HF_DATASETS[name]
    local_path = _local_dataset_path(filename)
    if local_path.exists():
        return local_path

    try:
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=DATA_DIR,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN", None),
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to download {name}: {exc}") from exc

    if not local_path.exists():
        raise RuntimeError(f"Dataset {name} failed to download (path missing)")
    return local_path


# Convenience wrappers used by `src.main`

def prepare_exp1_real() -> None:
    for ds in ("chameleon", "squirrel"):
        fetch_hf_dataset(ds)


def ensure_all_full() -> None:
    """Called once in full experiment before training starts."""
    prepare_exp1_real()
    # Additional datasets (Pokec, OGBN) rely on their native libraries and
    # are therefore fetched transparently inside those loaders.
