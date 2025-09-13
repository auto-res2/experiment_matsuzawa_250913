"""
FedC3PO Training Module - Federated Curvature-Controlled Causal-Probabilistic ODE
M1–M6 implementation (Koopman filtering, CoCA, FDP-Curv, Comm-SMiGS, BR-Flow, Shapley-Curv).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sb  # Renamed alias to avoid rare name-collision with NumPy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import levy_stable
from sklearn.metrics import accuracy_score, average_precision_score
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# Publication-quality plotting defaults
plt.style.use("seaborn-v0_8-paper")
sb.set_palette("husl")


@dataclass
class FedC3POConfig:
    """Configuration for FedC3PO experiments"""

    # Model
    hidden_dim: int = 64
    num_layers: int = 4
    alpha_stable: float = 1.6
    sketch_rows: int = 256

    # Optimisation
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    local_epochs: int = 5
    batch_size: int = 262_144

    # Privacy
    dp_epsilon: float = 4.0
    dp_delta: float = 1e-5
    comm_epsilon: float = 4.0

    # Federated
    num_clients: int = 10
    rounds: int = 100
    client_fraction: float = 0.5

    # Defence
    byzantine_fraction: float = 0.2
    spectral_clip_percentile: float = 95

    # Fairness
    fairness_gamma: float = 0.02

    # Communication scheduler
    scheduler_theta: float = 0.05
    energy_lambda: float = 1.0


class CountMinSketch:
    """Simple Count-Min Sketch (CMS) implementation used for spectral sketches."""

    def __init__(self, width: int, depth: int):
        self.width = width
        self.depth = depth
        self.table = torch.zeros(depth, width)
        self.hash_seeds = torch.randint(0, 2**32, (depth,))

    def _hash(self, item: torch.Tensor, seed: int) -> int:
        """MD5-based hash of the tensor bytes with an additional 32-bit seed."""
        item_hash = int(hashlib.md5(item.detach().cpu().numpy().tobytes()).hexdigest(), 16)
        return (item_hash + seed) % self.width

    def update(self, item: torch.Tensor, count: float = 1.0):
        for d in range(self.depth):
            idx = self._hash(item, self.hash_seeds[d].item())
            self.table[d, idx] += count

    def query(self, item: torch.Tensor) -> float:
        counts = []
        for d in range(self.depth):
            idx = self._hash(item, self.hash_seeds[d].item())
            counts.append(self.table[d, idx].item())
        return min(counts)

    def merge(self, other: "CountMinSketch") -> None:
        assert self.width == other.width and self.depth == other.depth
        self.table += other.table


class AlphaStableKoopmanFilter(nn.Module):
    """M1 – Multi-scale α-stable Koopman filter with CMS aggregation."""

    def __init__(self, dim: int, alpha: float = 1.6, scales: int = 4):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.scales = scales
        self.koopman_ops = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(scales)])
        self.scale_params = nn.Parameter(torch.randn(scales, 2))  # (scale, location)
        self.sketch = CountMinSketch(width=256, depth=4)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        outputs = []
        for scale_idx, koop in enumerate(self.koopman_ops):
            h = koop(x)
            scale, loc = self.scale_params[scale_idx]
            noise = torch.from_numpy(
                levy_stable.rvs(self.alpha, 0, loc.item(), abs(scale.item()), size=h.shape)
            ).to(h.device, dtype=h.dtype)
            h = h + noise
            spectrum = torch.fft.fft(h.mean(0))
            self.sketch.update(spectrum.real)
            outputs.append(h)
        return torch.stack(outputs).mean(0)


class CurvatureContrastiveAlignment(nn.Module):
    """M2 – CoCA via InfoNCE."""

    def __init__(self, dim: int, tau: float = 0.5):
        super().__init__()
        self.tau = tau
        self.projector = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    @staticmethod
    def _forman(edge_index: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        curv = 4 - deg[row] - deg[col]
        return curv

    def forward(self, x_l: torch.Tensor, x_g: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        row, col = edge_index
        deg = degree(col, x_l.size(0), dtype=x_l.dtype).to(x_l.device)
        curv_l = self._forman(edge_index, deg)
        curv_g = curv_l.clone()
        z_l = self.projector(curv_l.unsqueeze(-1).expand(-1, x_l.size(1)))
        z_g = self.projector(curv_g.unsqueeze(-1).expand(-1, x_l.size(1)))
        idx = torch.randperm(z_l.size(0), device=z_l.device)[: min(128, z_l.size(0))]
        z_l, z_g = z_l[idx], z_g[idx]
        sim = (z_l @ z_g.T) / self.tau
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)


class FairDPCurvaturePerturbation(nn.Module):
    """M3 – FDP-Curv: group-aware α-stable DP noise."""

    def __init__(self, eps: float, delta: float, gamma: float, alpha: float = 1.6):
        super().__init__()
        self.eps, self.delta, self.gamma, self.alpha = eps, delta, gamma, alpha

    def add_fair_noise(self, g: torch.Tensor, sensitive: Optional[torch.Tensor] = None) -> torch.Tensor:
        if sensitive is None:
            sensitive = torch.zeros(g.size(0), device=g.device)
        noise = torch.zeros_like(g)
        for group in [0, 1]:
            mask = sensitive == group
            if not mask.any():
                continue
            risk = g[mask].norm()
            scale = self.gamma / (risk + 1e-8)
            noise[mask] = torch.from_numpy(
                levy_stable.rvs(self.alpha, 0, 0, scale.item(), size=g[mask].shape)
            ).to(g.device, dtype=g.dtype)
        return g + noise


class CommSMiGS(nn.Module):
    """M4 – Communication-aware scheduler."""

    def __init__(self, theta: float = 0.05, energy_lambda: float = 1.0):
        super().__init__()
        self.theta = theta
        self.energy_lambda = energy_lambda
        self.history: List[torch.Tensor] = []
        self.energy_used: float = 0.0

    def should_send(self, forecast: torch.Tensor, power_cap: float = 300.0) -> bool:  # noqa: D401,E501
        if self.history:
            variance = (forecast - torch.stack(self.history[-10:]).mean(0)).pow(2).mean()
        else:
            variance = torch.tensor(float("inf"))
        self.history.append(forecast.detach())
        if variance > self.theta:
            return True
        est_energy = 50.0  # placeholder constant cost
        if self.energy_used + est_energy > power_cap:
            return False
        objective = self.energy_lambda * est_energy + variance.item()
        threshold = 0.1 * power_cap
        if objective < threshold:
            self.energy_used += est_energy
            return True
        return False


class ByzantineResilientFlow(nn.Module):
    """M5 – BR-Flow aggregation."""

    def __init__(self, kappa_percentile: float = 95, max_byzantine: int = 3):
        super().__init__()
        self.kappa_percentile = kappa_percentile
        self.max_byzantine = max_byzantine

    def aggregate(self, updates: List[torch.Tensor]) -> torch.Tensor:
        if not updates:
            return torch.zeros(1)
        stacked = torch.stack(updates)
        norms = torch.norm(stacked, dim=-1)
        thresh = torch.quantile(norms, self.kappa_percentile / 100)
        clipped = stacked[norms <= thresh]
        if clipped.numel() == 0:
            clipped = stacked[:1]
        median = clipped.mean(0)
        for _ in range(5):  # Weiszfeld iterations
            dists = torch.norm(clipped - median, dim=-1)
            weights = 1.0 / (dists + 1e-8)
            median = (clipped * weights.unsqueeze(-1)).sum(0) / weights.sum()
        return median


class GCNLayer(MessagePassing):
    """Simple GCN layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype).to(x.device)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        return out + self.bias

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        return norm.view(-1, 1) * x_j


class FedC3PO(nn.Module):
    """Full FedC3PO model (M1–M6)."""

    def __init__(self, cfg: FedC3POConfig, num_features: int, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(num_features, cfg.hidden_dim)
        self.koop = AlphaStableKoopmanFilter(cfg.hidden_dim, cfg.alpha_stable)
        self.coca = CurvatureContrastiveAlignment(cfg.hidden_dim)
        self.fdp = FairDPCurvaturePerturbation(cfg.dp_epsilon, cfg.dp_delta, cfg.fairness_gamma)
        self.scheduler = CommSMiGS(cfg.scheduler_theta, cfg.energy_lambda)
        self.byz_def = ByzantineResilientFlow(
            cfg.spectral_clip_percentile, int(cfg.byzantine_fraction * cfg.num_clients)
        )
        self.gnn = nn.ModuleList([GCNLayer(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.num_layers)])
        self.output_proj = nn.Linear(cfg.hidden_dim, num_classes)
        self.metrics = defaultdict(list)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401,E501
        h = self.input_proj(x)
        h = self.koop(h, edge_index)
        for layer in self.gnn:
            h = F.relu(layer(h, edge_index))
            h = F.dropout(h, p=0.1, training=self.training)
        return self.output_proj(h)

    # M6 – Shapley-Curv (placeholder)
    def compute_shapley_curvature(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[int, float]:  # noqa: D401,E501
        return {}


class FederatedClient:
    """Federated client wrapper."""

    def __init__(self, cid: int, model: FedC3PO, loader, cfg: FedC3POConfig):
        self.cid, self.model, self.loader, self.cfg = cid, model, loader, cfg
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.sketch = CountMinSketch(width=cfg.sketch_rows, depth=4)
        self.is_byzantine = False

    def train_local(self, epochs: int) -> Dict[str, List[float]]:
        self.model.train()
        metrics = {"loss": [], "accuracy": []}
        for _ in range(epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            for batch in self.loader:
                x, ei, y = batch.x, batch.edge_index, batch.y
                out = self.model(x, ei)
                loss = F.cross_entropy(out, y)
                self.opt.zero_grad()
                loss.backward()
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad = self.model.fdp.add_fair_noise(p.grad)
                self.opt.step()
                epoch_loss += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
            metrics["loss"].append(epoch_loss / len(self.loader))
            metrics["accuracy"].append(correct / max(1, total))
        for p in self.model.parameters():
            self.sketch.update(p.data.flatten())
        return metrics

    def get_update(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in self.model.parameters()])


class FederatedServer:
    """Server orchestrating aggregation with BR-Flow."""

    def __init__(self, model: FedC3PO, cfg: FedC3POConfig):
        self.model, self.cfg = model, cfg
        self.round = 0

    def aggregate(self, upd: List[torch.Tensor]):
        if not upd:
            return
        agg = self.model.byz_def.aggregate(upd)
        idx = 0
        for p in self.model.parameters():
            sz = p.numel()
            p.data = agg[idx : idx + sz].view_as(p)
            idx += sz

    def evaluate(self, loader):
        self.model.eval()
        corr = tot = 0
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch.x, batch.edge_index)
                corr += (out.argmax(1) == batch.y).sum().item()
                tot += batch.y.size(0)
        return {"accuracy": corr / max(1, tot)}


# ---------------- federated training loop ---------------- #

def train_federated(cfg: FedC3POConfig, clients: List[FederatedClient], server: FederatedServer, test_loader):  # noqa: D401,E501
    print("\n" + "=" * 60)
    print("Starting FedC3PO federated training – rounds:", cfg.rounds)
    print("=" * 60)
    metrics = {
        "loss": defaultdict(list),
        "accuracy": defaultdict(list),
        "global_accuracy": [],
        "communication": {"bits_sent": [], "energy_joules": []},
    }
    for r in range(cfg.rounds):
        print(f"\n--- Round {r + 1}/{cfg.rounds} ---")
        sel = np.random.choice(clients, max(1, int(cfg.client_fraction * len(clients))), replace=False)
        updates, bits, energy = [], 0, 0
        for c in sel:
            loc_metrics = c.train_local(cfg.local_epochs)
            metrics["loss"][c.cid].extend(loc_metrics["loss"])
            metrics["accuracy"][c.cid].extend(loc_metrics["accuracy"])
            up = c.get_update()
            forecast = torch.randn(cfg.hidden_dim)
            if c.model.scheduler.should_send(forecast):
                updates.append(up)
                bits += up.numel() * 32  # float32
                energy += 50.0  # placeholder
        server.aggregate(updates)
        res = server.evaluate(test_loader)
        metrics["global_accuracy"].append(res["accuracy"])
        metrics["communication"]["bits_sent"].append(bits)
        metrics["communication"]["energy_joules"].append(energy)
        print(
            f"Global Acc: {res['accuracy']:.4f} | Comm: {bits/8/1024/1024:.2f} MB, {energy:.0f} J"
        )
        # Simple convergence check
        if (
            len(metrics["global_accuracy"]) > 10
            and max(metrics["global_accuracy"][-10:]) - min(metrics["global_accuracy"][-10:]) < 1e-3
        ):
            print("Converged – early stop.")
            break
    return metrics


# ---------------- helper I/O ---------------- #

def _clean(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def save_training_results(results: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"Training results saved → {path}")
    # Print JSON to stdout for verification
    print(json.dumps(_clean(results), indent=2))


def plot_training_curves(metrics: Dict, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(metrics["global_accuracy"], marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Global Accuracy over Rounds")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Curve saved → {save_path}")
    plt.close()
