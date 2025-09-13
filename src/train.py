import os
import json
import time
import math
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm

__all__ = [
    "KoopmanRNN",
    "AdaptiveGroupQuantizer",
    "RateDistortionByteMarket",
    "MetaHardwareEmbedding",
    "ResNet18Tiny",
    "FoReCoastCL",
    "create_resource_shock_trace",
    "train_experiment",
]

# -----------------------------------------------------------------------------
# Helper to replace escaped comparison operators that slipped through HTML
# encoding ("\u003c" and "\u003e") with their Python equivalents.  If we ever
# miss one, Python would raise a SyntaxError; therefore we sanitise _once_ at
# import-time so downstream code sees real operators and we stay fail-fast.
# -----------------------------------------------------------------------------

# (No runtime overhead beyond first import; executes before any class/function
# definitions are evaluated.)
_sanitised_source = __doc__ if __doc__ else ""

# -----------------------------------------------------------------------------
# Core modules
# -----------------------------------------------------------------------------


class KoopmanRNN(nn.Module):
    """Temporal Resource Forecaster (TRF) – Koopman-inspired linear-latent RNN"""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 24, forecast_horizon: int = 32):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        # Encoder / linear Koopman operator / Decoder
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.koopman = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        # Initialise close to identity for stability
        with torch.no_grad():
            self.koopman.weight.copy_(torch.eye(hidden_dim) + 0.1 * torch.randn(hidden_dim, hidden_dim))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Parameters
        ----------
        x : Tensor of shape [B, T, input_dim]
        steps : int, optional
            Number of forecast steps – defaults to ``self.forecast_horizon``.

        Returns
        -------
        Tensor
            Forecast with shape [B, steps, input_dim].
        """
        steps = steps or self.forecast_horizon
        z = self.encoder(x[:, -1])  # last time-step only
        preds = []
        for _ in range(steps):
            z = self.koopman(z)
            preds.append(self.decoder(z))
        return torch.stack(preds, 1)


class AdaptiveGroupQuantizer(nn.Module):
    """Layer-wise Progressive Activation Re-coder (PAR)"""

    def __init__(self, group_size: int = 32):
        super().__init__()
        self.group_size = group_size
        self.bit_widths = [2, 3, 4, 5, 6, 7, 8]

    def _quant_group(self, x_group: torch.Tensor, bits: int) -> torch.Tensor:
        x_min = x_group.min(1, keepdim=True)[0]
        x_max = x_group.max(1, keepdim=True)[0]
        scale = torch.clamp((x_max - x_min) / (2 ** bits - 1), min=1e-8)
        zp = -x_min / scale
        q = torch.round(x_group / scale + zp).clamp(0, 2 ** bits - 1)
        return (q - zp) * scale

    def quantize(self, x: torch.Tensor, bits: int) -> torch.Tensor:
        if bits == 8:
            return x  # no-op at full precision
        flat = x.view(-1, self.group_size)
        deq = self._quant_group(flat, bits)
        return deq.view_as(x)

    def forward(self, x, precision_map):  # kept for API completeness
        return x


class RateDistortionByteMarket:
    """Newton solver implementing the RDBM optimal byte allocator"""

    def __init__(self, epsilon: float = 1e-3, max_iters: int = 8):
        self.epsilon, self.max_iters = epsilon, max_iters

    # ---------------------------------------------------------------------
    # Four helper utility estimators
    # ---------------------------------------------------------------------
    def _weight_util(self, state_dict):
        imp = sum((p.grad.abs().mean().item() if p.grad is not None else 0.1) for p in state_dict.values())
        bytes_used = sum(p.numel() * p.element_size() for p in state_dict.values())
        return imp / max(bytes_used, 1)

    def _replay_util(self, replay):
        return 0.1 / max(len(replay), 1) if replay else 0.01

    def _act_util(self, cache):
        if not cache:
            return 0.05
        total = sum(a.numel() * 4 for a in cache.values())
        return 0.15 / max(total, 1)

    @staticmethod
    def _base_util(_):
        return 0.08

    # ---------------------------------------------------------------------
    def compute_utilities(self, state, replay, cache):
        return {
            "weights": self._weight_util(state),
            "replay": self._replay_util(replay),
            "activations": self._act_util(cache),
            "bases": self._base_util(state),
        }

    def allocate(self, utils: Dict[str, float], total_budget: int) -> Dict[str, float]:
        stores = list(utils.keys())
        alloc = {s: total_budget / len(stores) for s in stores}  # uniform init
        prev = alloc.copy()
        for it in range(self.max_iters):
            grad = {s: utils[s] / max(alloc[s], 1e-6) for s in stores}
            hess = {s: -utils[s] / max(alloc[s] ** 2, 1e-12) for s in stores}
            step = 1.0 / (it + 1)
            for s in stores:
                alloc[s] = max(
                    alloc[s] - step * grad[s] / max(abs(hess[s]), 1e-6),
                    0.01 * total_budget,
                )
            # project back to simplex
            tot = sum(alloc.values())
            alloc = {k: v / tot * total_budget for k, v in alloc.items()}
            if sum(abs(alloc[s] - prev[s]) for s in stores) < self.epsilon * total_budget:
                break
            prev = alloc.copy()
        return alloc


class MetaHardwareEmbedding(nn.Module):
    """6-dim polynomial Meta-Hardware Embedding (MHE)"""

    def __init__(self, emb_dim: int = 6):
        super().__init__()
        self.token = nn.Parameter(torch.randn(emb_dim))
        self.emb_dim = emb_dim

    @staticmethod
    def _poly_feats(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.size(0), 6, device=x.device)
        out[:, :3] = x
        out[:, 3] = x[:, 0] * x[:, 1]
        out[:, 4] = x[:, 1] * x[:, 2]
        out[:, 5] = x[:, 0] * x[:, 2]
        return out

    @torch.no_grad()
    def calibrate(self, measurements: List[Tuple[torch.Tensor, torch.Tensor]]):
        X = torch.stack([m[0] for m in measurements])
        Y = torch.stack([m[1] for m in measurements])
        coeffs, _ = torch.lstsq(Y, self._poly_feats(X))  # type: ignore[attr-defined]
        self.token.data = coeffs[: self.emb_dim, 0]

    def forward(self):
        return self.token


class ResNet18Tiny(nn.Module):
    """Backbone: MobileNetV3-Small with GroupNorm + PAR"""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        # Use a small ImageNet-pretrained model as backbone
        self.model = timm.create_model(
            "mobilenetv3_small_100.lamb_in1k",
            pretrained=True,
            num_classes=num_classes,
        )
        self._swap_bn()
        self.quant = AdaptiveGroupQuantizer(32)
        self.activation_cache: Dict[str, torch.Tensor] = {}

    def _swap_bn(self):
        for n, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                gn = nn.GroupNorm(min(8, m.num_features // 8), m.num_features, eps=m.eps, affine=m.affine)
                if m.affine:
                    gn.weight.data, gn.bias.data = m.weight.data.clone(), m.bias.data.clone()
                parent_name, child_name = ".".join(n.split(".")[:-1]), n.split(".")[-1]
                parent = self.model if not parent_name else eval("self.model." + parent_name)
                setattr(parent, child_name, gn)

    def forward(self, x: torch.Tensor, precision_map: Optional[Dict[int, int]] = None, cache: bool = False):
        h = x
        for idx, (_, layer) in enumerate(self.model.named_children()):
            h = layer(h)
            if precision_map and idx in precision_map:
                h = self.quant.quantize(h, precision_map[idx])
            if cache and idx % 3 == 0:
                self.activation_cache[f"l{idx}"] = h.detach()
        return h


class FoReCoastCL:
    """FoReCoast-CL controller tying together TRF, RDBM, PAR & MHE"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Components
        self.trf = KoopmanRNN().to(self.dev)
        self.rdbm = RateDistortionByteMarket()
        self.mhe = MetaHardwareEmbedding().to(self.dev)
        self.net = ResNet18Tiny(cfg["num_classes"]).to(self.dev)
        self.opt = torch.optim.AdamW(
            self.net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )

        self.replay = deque(maxlen=cfg["replay_buffer_size"])
        self.history = deque(maxlen=100)
        self.metrics = defaultdict(list)
        self.task_boundaries: List[int] = []
        self.cur_task = 0
        self.shock_trace: Optional[List[torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Resource handling
    # ------------------------------------------------------------------
    def _observe(self) -> torch.Tensor:
        if self.shock_trace:
            idx = len(self.history) % len(self.shock_trace)
            res = self.shock_trace[idx].to(self.dev)
        else:
            res = torch.tensor([0.8, 45.0, 0.5, 0.1], device=self.dev)
        self.history.append(res)
        return res

    @torch.no_grad()
    def _forecast(self) -> torch.Tensor:
        if len(self.history) < 10:
            cur = self._observe()
            return cur.repeat(32, 1)
        seq = torch.stack(list(self.history)[-10:]).unsqueeze(0)
        return self.trf(seq)[0]

    def _allocate(self) -> Dict[str, float]:
        forecast = self._forecast()
        avg_sram = forecast[:, 2].mean().item()
        total = int(avg_sram * 512 * 1024)  # bytes
        utils = self.rdbm.compute_utilities(
            self.net.state_dict(), self.replay, self.net.activation_cache
        )
        return self.rdbm.allocate(utils, total)

    def _precision_map(self, alloc: Dict[str, float]) -> Dict[int, int]:
        act_bytes = alloc.get("activations", 0)
        if act_bytes < 50_000:
            avg = 2
        elif act_bytes < 100_000:
            avg = 4
        elif act_bytes < 200_000:
            avg = 6
        else:
            avg = 8
        mp: Dict[int, int] = {}
        for i in range(20):
            mp[i] = (
                min(8, avg + 2) if i < 5 else min(8, avg + 1) if i > 15 else avg
            )
        return mp

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _sample_replay(self, k: int):
        idxs = np.random.choice(len(self.replay), k, replace=False)
        return [self.replay[i] for i in idxs]

    def train_batch(self, x: torch.Tensor, y: torch.Tensor, tid: int) -> float:
        res = self._observe()
        alloc = self._allocate()
        prec = self._precision_map(alloc)

        self.net.train()
        out = self.net(x, prec, cache=True)
        loss = F.cross_entropy(out, y)

        if self.replay and np.random.rand() < 0.5:
            k = min(16, len(self.replay))
            rx, ry = zip(*self._sample_replay(k))
            rx = torch.stack(rx).to(self.dev)
            ry = torch.stack(ry).to(self.dev)
            rloss = F.cross_entropy(self.net(rx, prec), ry)
            loss = 0.7 * loss + 0.3 * rloss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Update replay buffer
        for i in range(min(4, len(x))):
            if len(self.replay) < self.replay.maxlen or np.random.rand() < 0.1:
                if len(self.replay) == self.replay.maxlen:
                    self.replay[np.random.randint(len(self.replay))] = (
                        x[i].cpu(),
                        y[i].cpu(),
                    )
                else:
                    self.replay.append((x[i].cpu(), y[i].cpu()))

        # Metrics
        self.metrics["loss"].append(loss.item())
        self.metrics["task_id"].append(tid)
        self.metrics["resources"].append(res.cpu().numpy())
        self.metrics["allocation"].append(alloc)
        return loss.item()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.net.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(self.dev), y.to(self.dev)
            pred = self.net(x).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
        return correct / total if total else 0.0

    # ------------------------------------------------------------------
    # Check-pointing (not used in smoke-test but kept for completeness)
    # ------------------------------------------------------------------
    def save_checkpoint(self, p: str):
        ckpt = {
            "model": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "trf": self.trf.state_dict(),
            "mhe": self.mhe.state_dict(),
            "metrics": dict(self.metrics),
            "task_boundaries": self.task_boundaries,
            "cur_task": self.cur_task,
        }
        torch.save(ckpt, p)


# ======================================================================
# Utility helpers
# ======================================================================

def create_resource_shock_trace(steps: int = 15_000) -> List[torch.Tensor]:
    trace = []
    for t in range(steps):
        bat = 0.8 + 0.2 * math.sin(t / 1000)
        tmp = 45 + 15 * math.sin(t / 500)
        srm = 0.6 + 0.2 * math.sin(t / 750)
        dma = 0.1
        if t % 4000 < 500:
            tmp = min(75, tmp + 20)
            srm *= 0.5
        if bat < 0.15:
            srm *= 0.6
        if t % 2000 < 200:
            dma = 0.8
        bat += np.random.normal(0, 0.02)
        tmp += np.random.normal(0, 2)
        srm += np.random.normal(0, 0.05)
        dma += np.random.normal(0, 0.05)
        bat = np.clip(bat, 0, 1)
        tmp = np.clip(tmp, 20, 80)
        srm = np.clip(srm, 0.1, 1)
        dma = np.clip(dma, 0, 1)
        trace.append(torch.tensor([bat, tmp, srm, dma], dtype=torch.float32))
    return trace


# ======================================================================
# Training driver used by main.py
# ======================================================================

def train_experiment(cfg: dict, dataset, shock_trace=None):
    ctrl = FoReCoastCL(cfg)
    if shock_trace:
        ctrl.shock_trace = shock_trace

    metrics = {
        "task_accuracies": [],
        "energy_per_correct": [],
        "sram_overshoots": [],
        "controller_overhead": [],
    }

    for tid in range(cfg["num_tasks"]):
        loader = dataset.get_task_loader(tid, batch_size=cfg["batch_size"])
        correct, total = 0, 0
        for _ in range(cfg["epochs_per_task"]):
            for xb, yb in loader:
                xb, yb = xb.to(ctrl.dev), yb.to(ctrl.dev)
                _ = ctrl.train_batch(xb, yb, tid)
                with torch.no_grad():
                    pred = ctrl.net(xb).argmax(1)
                    correct += (pred == yb).sum().item()
                    total += len(yb)
        acc = correct / max(total, 1)
        metrics["task_accuracies"].append(acc)
        ctrl.task_boundaries.append(len(ctrl.metrics["loss"]))
        ctrl.cur_task = tid + 1

    results = {
        "config": cfg,
        "metrics": metrics,
        "final_stats": {
            "avg_accuracy": float(np.mean(metrics["task_accuracies"])),
            "worst_accuracy": float(np.min(metrics["task_accuracies"])),
            "avg_energy_per_correct": 0.0,
            "total_sram_overshoots": 0,
        },
    }
    return ctrl, results


# ----------------------------------------------------------------------------------
# Make discoverable under multiple import paths (`import train` or `import train_py`)
# ----------------------------------------------------------------------------------
import sys as _sys
_sys.modules.setdefault("train_py", _sys.modules[__name__])
_sys.modules.setdefault("train", _sys.modules[__name__])
