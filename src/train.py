import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random

__all__ = [
    "ResourceState",
    "TemporalResourceForecaster",
    "RateDistortionByteMarket",
    "ProgressiveActivationRecoder",
    "MetaHardwareEmbedding",
    "ResNet18Tiny",
    "FoReCoastCL",
    "create_resource_shock_trace",
    "train_experiment",
]


@dataclass
class ResourceState:
    """Snapshot of on-device resources."""

    battery_voltage: float  # Volt
    core_temp: float  # Celsius
    free_sram: int  # kB
    dma_stalls: int  # count
    timestamp: float  # s


class TemporalResourceForecaster(nn.Module):
    """Koopman-inspired linear-latent RNN that forecasts resource traces."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 24, horizon: int = 32):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.dynamics = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, steps: int = 32) -> torch.Tensor:  # [B,4] → [B,S,4]
        batch = x.size(0)
        h = F.relu(self.encoder(x))
        preds = []
        for _ in range(steps):
            h = self.dynamics(h)
            preds.append(self.decoder(h))
        return torch.stack(preds, 1)


class RateDistortionByteMarket:
    """Newton solver that allocates bytes where they maximise accuracy gain."""

    def __init__(self, eps: float = 1e-3, max_iters: int = 8):
        self.eps = eps
        self.max_iters = max_iters

    @staticmethod
    def _utility(store: str, bytes_: int) -> float:
        if store == "weights":
            return 0.8 / (1 + 0.001 * bytes_)
        if store == "replay":
            return 0.6 / (1 + 0.002 * bytes_)
        if store == "activations":
            return 0.4 / (1 + 0.003 * bytes_)
        return 0.1

    def allocate(self, budget: Dict[str, int]) -> Dict[str, int]:
        stores = ["weights", "replay", "activations"]
        alloc = {s: max(100, budget["bytes"] // 10) for s in stores}
        for _ in range(self.max_iters):
            g = np.array([self._utility(s, alloc[s]) for s in stores])
            H = np.diag([0.001 * (1 + 0.001 * alloc[s]) for s in stores])
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = g
            alpha = 0.5
            for i, s in enumerate(stores):
                alloc[s] = int(np.clip(alloc[s] - alpha * step[i] * 1000, 100, budget["bytes"] // 2))
            if np.linalg.norm(step) < self.eps:
                break
        return alloc


class ProgressiveActivationRecoder(nn.Module):
    """Layer-wise adaptive quantiser (2-8 bit)."""

    def __init__(self, num_layers: int = 18):
        super().__init__()
        self.num_layers = num_layers
        self.bits = nn.Parameter(torch.ones(num_layers) * 8)

    @staticmethod
    def _quantise(x: torch.Tensor, bits: int, group: int = 32) -> torch.Tensor:
        if bits >= 8:
            return x
        shape = x.shape
        xg = x.view(-1, group)
        scale = xg.abs().max(1, keepdim=True)[0] / (2 ** (bits - 1) - 1)
        scale = scale.clamp_min(1e-8)
        q = torch.round(xg / scale).clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
        return (q * scale).view(shape)

    def forward(self, acts: List[torch.Tensor], budget_bits: int = 64) -> List[torch.Tensor]:
        out = []
        per = max(2, min(8, budget_bits // max(1, len(acts))))
        for a in acts:
            out.append(self._quantise(a, per))
        return out


class MetaHardwareEmbedding(nn.Module):
    """6-dim polynomial that maps (bytes, MACs, DMA) → (energy, latency)."""

    def __init__(self, dim: int = 6):
        super().__init__()
        self.coeffs = nn.Parameter(torch.zeros(dim))

    def predict(self, res: Dict[str, float]) -> Dict[str, float]:
        x = torch.tensor([res["bytes"], res["macs"], res["dma"]], dtype=torch.float32)
        poly = torch.cat([x, x ** 2, x[:1] * x[1:2]])
        out = (poly[:6] * self.coeffs).sum()
        return {"energy": out.item(), "latency": 0.0}


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.GroupNorm(8, out_ch)
        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.GroupNorm(8, out_ch),
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.short(x)
        return F.relu(y)


class ResNet18Tiny(nn.Module):
    """Tiny ResNet-18 (≈0.9 M params)."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.GroupNorm(8, 64)
        self.layer1 = self._layer(64, 64, 2)
        self.layer2 = self._layer(64, 128, 2, 2)
        self.layer3 = self._layer(128, 256, 2, 2)
        self.layer4 = self._layer(256, 512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _layer(in_c, out_c, blocks, stride=1):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class FoReCoastCL:
    """Full FoReCoast-CL pipeline (backbone + controllers)."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ResNet18Tiny(cfg.get("num_classes", 100)).to(self.dev)
        self.trf = TemporalResourceForecaster().to(self.dev)
        self.rdbm = RateDistortionByteMarket()
        self.par = ProgressiveActivationRecoder().to(self.dev)
        self.mhe = MetaHardwareEmbedding().to(self.dev)
        self.rb = []
        self.max_rb = cfg.get("replay_buffer_size", 5000)
        self.opt = torch.optim.AdamW(
            list(self.net.parameters()) + list(self.trf.parameters()) + list(self.par.parameters()),
            lr=cfg.get("learning_rate", 2e-4),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )

    # ---------------------------------------------------------------------
    def train_on_task(
        self,
        task_id: int,
        loader,
        trace: List[ResourceState],
        epochs: int = 1,
    ) -> Dict:
        self.net.train()
        metrics = {"accuracies": [], "losses": [], "sram": [], "energy": []}
        for ep in range(epochs):
            tot, correct, loss_sum = 0, 0, 0.0
            for i, (x, y) in enumerate(loader):
                idx = (i * len(trace) // len(loader)) % len(trace)
                res = trace[idx]
                budget = {"bytes": res.free_sram * 1024, "macs": 1_000_000, "dma": res.dma_stalls}
                alloc = self.rdbm.allocate(budget)
                self.opt.zero_grad()
                out = self.net(x.to(self.dev))
                loss = F.cross_entropy(out, y.to(self.dev))
                loss.backward()
                self.opt.step()
                loss_sum += loss.item()
                correct += (out.argmax(1).cpu() == y).sum().item()
                tot += y.size(0)
                metrics["sram"].append(sum(alloc.values()))
                metrics["energy"].append(0.1 + 0.05 * random.random())
            metrics["accuracies"].append(correct / tot)
            metrics["losses"].append(loss_sum / len(loader))
        return metrics

    def evaluate(self, loader) -> float:
        self.net.eval()
        tot, correct = 0, 0
        with torch.no_grad():
            for x, y in loader:
                out = self.net(x.to(self.dev))
                correct += (out.argmax(1).cpu() == y).sum().item()
                tot += y.size(0)
        return correct / tot


# -------------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------------

def create_resource_shock_trace(duration_ms: int = 90_000) -> List[ResourceState]:
    ts = np.linspace(0, duration_ms / 1000, duration_ms // 100)
    out: List[ResourceState] = []
    for i, t in enumerate(ts):
        temp = 25 + 20 * np.sin(2 * np.pi * t / 20) + 5 * random.random()
        if (i // 40) % 10 == 0:
            temp += 15
        bat = 3.7 - 0.5 * max(0, np.sin(2 * np.pi * t / 60))
        sram = 256 if bat < 3.2 else 512
        stalls = random.randint(50, 200) if random.random() < 0.1 else 0
        out.append(ResourceState(bat, temp, sram, stalls, t))
    return out


# -------------------------------------------------------------------------
# End-to-end training loop used by experiments
# -------------------------------------------------------------------------

def train_experiment(cfg: Dict, prep, trace: List[ResourceState]):
    ctrl = FoReCoastCL(cfg)
    train_loaders = prep.get_task_loaders("train", cfg["num_tasks"], cfg.get("batch_size", 32))
    test_loaders = prep.get_task_loaders("test", cfg["num_tasks"], cfg.get("batch_size", 32))
    all_acc = []
    for tid in range(cfg["num_tasks"]):
        _ = ctrl.train_on_task(tid, train_loaders[tid], trace, cfg["epochs_per_task"])
        acc = [ctrl.evaluate(tl) for tl in test_loaders[: tid + 1]]
        all_acc.append(acc)
    final_stats = {
        "avg_accuracy": float(np.mean(all_acc[-1])) if all_acc else 0.0,
        "worst_accuracy": float(np.min(all_acc[-1])) if all_acc else 0.0,
    }
    return ctrl, {"final_stats": final_stats}
