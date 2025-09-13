"""
Federated data preparation for real & simulated datasets.
Adds PyTorch <2.6 compatibility patch (weights_only=False) when loading OGB
pre-processed files and switches to the up-to-date torch_geometric.loader
DataLoader API. All dataset artefacts are stored under ./data.
"""

from __future__ import annotations

import os
import functools
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # modern API (avoids deprecation)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected

from huggingface_hub import hf_hub_download
from ogb.nodeproppred import NodePropPredDataset

HF_TOKEN = os.getenv("HF_TOKEN")  # optional auth for private HF resources

# -----------------------------------------------------------------------------
# Helper ───────────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def _patch_torch_load_for_ogb() -> None:
    """Monkey-patch torch.load so that *weights_only=False* is passed by default.

    PyTorch ≥2.6 changed the default value of ``weights_only`` to True which
    breaks deserialisation of OGB's pre-processed *.pt* files. We inject a tiny
    wrapper **once** to restore the historic behaviour *only* when no explicit
    argument is provided.
    """

    if getattr(torch, "_fedc3po_torch_load_patched", False):
        return  # already patched in this interpreter session

    original_load = torch.load

    @functools.wraps(original_load)
    def _compat_load(*args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _compat_load  # type: ignore[assignment]
    torch._fedc3po_torch_load_patched = True  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Main dataset class ──────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

class FederatedGraphDataset:
    def __init__(self, dataset_name: str, num_clients: int = 3, root: str = "./data", seed: int = 42):
        self.name, self.num_clients = dataset_name, num_clients
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.data = self._load()
        self.client_data = self._partition()

    # ---------------------------- data loading ----------------------------- #
    def _load(self) -> Data:
        if self.name == "ogbn-arxiv":
            # Patch torch.load <--> OGB incompatibility *before* the dataset is
            # instantiated (it triggers torch.load internally).
            _patch_torch_load_for_ogb()

            ds = NodePropPredDataset(name="ogbn-arxiv", root=str(self.root))
            g, y = ds[0]
            edge_index = torch.from_numpy(g["edge_index"])
            x = torch.from_numpy(g["node_feat"]).float()
            y = torch.from_numpy(y).view(-1)
            edge_index = to_undirected(edge_index)
            split = ds.get_idx_split()
            m_train = torch.zeros(x.size(0), dtype=torch.bool)
            m_train[split["train"]] = True
            m_val = torch.zeros_like(m_train)
            m_val[split["valid"]] = True
            m_test = torch.zeros_like(m_train)
            m_test[split["test"]] = True
            return Data(
                x=x,
                edge_index=edge_index,
                y=y,
                train_mask=m_train,
                val_mask=m_val,
                test_mask=m_test,
            )

        # ------------------------------------------------------------------ #
        # Fallback: synthetic toy graph (used by smoke-test for other names)
        # ------------------------------------------------------------------ #
        n, e, f, c = 500, 2_000, 32, 3
        x = torch.randn(n, f)
        edge_index = torch.randint(0, n, (2, e))
        y = torch.randint(0, c, (n,))
        m_train = torch.zeros(n, dtype=torch.bool)
        m_test = torch.zeros_like(m_train)
        m_train[: int(0.8 * n)] = True
        m_test[int(0.8 * n) :] = True
        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=m_train,
            test_mask=m_test,
            val_mask=m_test.clone(),
        )

    # --------------------------- partitioning ------------------------------ #
    def _partition(self) -> List[Data]:
        print(f"Creating {self.num_clients} federated partitions for {self.name}")
        train_nodes = self.data.train_mask.nonzero(as_tuple=True)[0]
        per_client = len(train_nodes) // self.num_clients
        perm = train_nodes[torch.randperm(len(train_nodes))]
        subsets: List[Data] = []
        for cid in range(self.num_clients):
            s, e = cid * per_client, (cid + 1) * per_client if cid < self.num_clients - 1 else len(train_nodes)
            mask_train = torch.zeros_like(self.data.train_mask)
            mask_train[perm[s:e]] = True
            subsets.append(
                Data(
                    x=self.data.x.clone(),
                    edge_index=self.data.edge_index.clone(),
                    y=self.data.y.clone(),
                    train_mask=mask_train,
                    val_mask=self.data.val_mask.clone(),
                    test_mask=self.data.test_mask.clone(),
                )
            )
        return subsets

    # ---------------------------- loaders ---------------------------------- #
    def get_client_loader(self, cid: int) -> DataLoader:
        return DataLoader([self.client_data[cid]], batch_size=1)

    def get_test_loader(self) -> DataLoader:
        # Return the *full* graph to avoid index issues during message passing
        return DataLoader([self.data], batch_size=1)


# Convenience wrapper (used by external modules) --------------------------------

def prepare_federated_data(cfg: dict) -> Tuple[List[DataLoader], DataLoader]:
    ds = FederatedGraphDataset(cfg.get("dataset", "ogbn-arxiv"), cfg.get("num_clients", 3))
    loaders = [ds.get_client_loader(i) for i in range(cfg.get("num_clients", 3))]
    return loaders, ds.get_test_loader()
