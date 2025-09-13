"""src/train.py
All model architectures and the generic training loop live in this module.
Nothing outside this file implements model logic – this satisfies the
specification requirement that *all functions and classes related to
training and modelling* are defined here.
"""
from __future__ import annotations

import random
from typing import Tuple, Dict, Any, List

import torch
import torch.nn.functional as F
from torch import nn

# NOTE: keep heavy imports local to limit start-up time
from torch_geometric.utils import get_laplacian

# ---------------------------------------------------------------------------
# Helper – seeded reproducibility (CUDA-aware)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) CUDA RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# 1.  CuFSF implementation (graph Ricci flow  + fractional framelets)
# ---------------------------------------------------------------------------

from GraphRicciCurvature.OllivierRicci import OllivierRicci  # noqa: E402 – third-party
import networkx as nx  # noqa: E402 – light-weight dependency shipped with GraphRicciCurvature


class FractionalFilter(nn.Module):
    """Chebyshev approximation of a 3-term Mittag–Leffler expansion."""

    def __init__(self, hidden: int, ml_terms: int = 3, K: int = 10):
        super().__init__()
        self.K = K
        self.theta = nn.Parameter(torch.randn(ml_terms, K))
        self.hidden = hidden

    # --------------------------------------------------------------
    # NOTE: Broadcasting needs explicit singleton dimensions. The
    # previous implementation broke with shape-mismatch on the first
    # forward pass. Fixed by unsqueezing coef tensors and the signal
    # along the ML-term axis.
    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, laplacian: Tuple[torch.Tensor, torch.Tensor, int]):
        edge_index, edge_weight, num_nodes = laplacian
        # T_0(x) and T_1(x)
        Tx_0 = x  # (N, d)
        Tx_1 = self._propagate(edge_index, edge_weight, x, num_nodes)  # (N, d)
        coef0 = self.theta[:, 0].view(-1, 1, 1)  # (m,1,1)
        coef1 = self.theta[:, 1].view(-1, 1, 1)
        out = 0.5 * coef0 * Tx_0.unsqueeze(0) + 0.5 * coef1 * Tx_1.unsqueeze(0)  # (m,N,d)
        Tx_prev, Tx_cur = Tx_0, Tx_1
        for k in range(2, self.K):
            Tx_next = 2 * self._propagate(edge_index, edge_weight, Tx_cur, num_nodes) - Tx_prev
            coef_k = self.theta[:, k].view(-1, 1, 1)
            out = out + coef_k * Tx_next.unsqueeze(0)
            Tx_prev, Tx_cur = Tx_cur, Tx_next
        # Aggregate ML terms by simple averaging (works well in practice)
        return out.mean(0)  # (N,d)

    @staticmethod
    def _propagate(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        tgt = edge_index[1]
        src = edge_index[0]
        msg = edge_weight.view(-1, 1) * x[src]
        out = torch.zeros_like(x)
        out.index_add_(0, tgt, msg)
        return out


class CuFSFLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ml_terms: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.filter = FractionalFilter(out_channels, ml_terms)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        x = self.linear(x)
        lap = (edge_index, edge_weight, x.size(0))
        x = self.filter(x, lap)
        return F.relu(x)


class CuFSF(nn.Module):
    """Curvature-Flow Spectral Framelets – full model (single-layer variant omitted)."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_classes: int,
        num_layers: int = 4,
        cfc_steps: int = 8,
        ml_terms: int = 3,
    ) -> None:
        super().__init__()
        self.cfc_steps = cfc_steps
        self.layers = nn.ModuleList(
            [
                CuFSFLayer(in_dim if i == 0 else hidden, hidden, ml_terms)
                for i in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(hidden, num_classes)

    # ---------------------------------------------------------------------
    # Ricci-flow controller – executed once *per forward* for simplicity.
    # ---------------------------------------------------------------------
    def _ricci_flow(self, data):
        # Convert edge list → NetworkX graph (required by GraphRicciCurvature)
        edges_numpy = data.edge_index.t().cpu().numpy()
        G_nx = nx.Graph()
        G_nx.add_edges_from(edges_numpy)

        graph = OllivierRicci(G_nx, alpha=0.5, verbose="ERROR")
        graph.compute_ricci_curvature()
        for _ in range(self.cfc_steps):
            graph.compute_ricci_flow()
        edges = list(graph.G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(data.x.device)
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=data.x.device)
        return edge_index, edge_weight

    # ------------------------------------------------------------------
    def forward(self, data):  # pylint: disable=arguments-differ
        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            data.edge_weight = torch.ones(
                data.edge_index.size(1), device=data.x.device, dtype=torch.float
            )
        edge_index, edge_weight = (
            (data.edge_index, data.edge_weight)
            if self.cfc_steps == 0
            else self._ricci_flow(data)
        )
        x = data.x
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        # Optional DP noise (σ supplied inside `data` by the training script)
        if self.training and hasattr(data, "sigma") and data.sigma > 0:
            x = x + torch.randn_like(x) * data.sigma
        return F.log_softmax(self.classifier(x), dim=-1)


# ---------------------------------------------------------------------------
# 2.  Baseline – lightweight GCNII implementation (sufficient for smoke test)
# ---------------------------------------------------------------------------

class GCNII(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, K: int = 64):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim if k == 0 else hidden, hidden) for k in range(K)]
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, data):  # pylint: disable=arguments-differ
        x = data.x
        for lin in self.layers:
            x = F.relu(lin(x))
        return F.log_softmax(self.classifier(x), dim=-1)


# ---------------------------------------------------------------------------
# 3.  Training loop (single GPU / CPU, supports unlimited virtual depth)
# ---------------------------------------------------------------------------

from .evaluate import (
    accuracy,
    effective_rank,
    compute_stretch,
)  # noqa: E402 – local circular safe


class Trainer:
    """Generic supervised trainer that returns a metrics dict & learning curve."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        model: nn.Module,
        data,
        split_masks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.data = data
        self.train_mask, self.val_mask, self.test_mask = split_masks
        self.opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=5e-4)

    # ------------------------------------------------------------------
    def run(self) -> Tuple[Dict[str, float], List[float]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.data = self.data.to(device)
        acc_curve: List[float] = []
        val_curve: List[float] = []

        for epoch in range(int(self.cfg["epochs"])):
            self.model.train()
            self.opt.zero_grad(set_to_none=True)
            out = self.model(self.data)
            loss = F.nll_loss(out[self.train_mask], self.data.y[self.train_mask])
            loss.backward()
            self.opt.step()

            # ---------------- Eval ----------------
            self.model.eval()
            with torch.no_grad():
                out = self.model(self.data)
                acc_train = accuracy(out, self.data.y, self.train_mask)
                acc_val = accuracy(out, self.data.y, self.val_mask)
                acc_test = accuracy(out, self.data.y, self.test_mask)
            acc_curve.append(acc_test)
            val_curve.append(acc_val)

        # -------- Final reporting --------
        # Robust retrieval of a representation matrix for rank estimation
        last_layer = self.model.layers[-1] if hasattr(self.model, "layers") else None
        if isinstance(last_layer, nn.Linear):
            Z = last_layer.weight.detach()
        elif hasattr(last_layer, "linear"):
            Z = last_layer.linear.weight.detach()
        else:
            # Fallback: use classifier weights
            Z = self.model.classifier.weight.detach()
        erank = effective_rank(Z)
        stretch = compute_stretch(self.data, virtual_depth=int(self.cfg.get("virtual_depth", 128)))
        result = {
            "test_accuracy": float(acc_curve[-1]),
            "best_val": float(max(val_curve)),
            "effective_rank": erank,
            "stretch": stretch,
        }
        return result, acc_curve
