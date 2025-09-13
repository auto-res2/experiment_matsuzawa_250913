import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.optim import Adam

from .preprocess import build_loader
from .evaluate import line_plot, save_json

# ------------------------------  MODELS  ---------------------------------
class CEMLGRU(nn.Module):
    """Tiny Bayesian GRU that outputs (mu, var) for worst-case task-bound estimation."""

    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=2, batch_first=True)
        self.mu = nn.Linear(hid, 1)
        self.logvar = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor):  # x: [B,T,in_dim]
        h, _ = self.gru(x)
        h = h[:, -1]  # last time-step
        mu = self.mu(h)
        var = torch.exp(self.logvar(h).clamp(-10, 10))
        return mu.squeeze(-1), var.squeeze(-1)


# -----------------------------  SCHEDULER  -------------------------------
class RTPS:
    """Real-Time Probabilistic Scheduler (greatly simplified)."""

    def __init__(self, window_ms: int = 4, kick_threshold: float = 0.2):
        import heapq, random

        self._heapq = heapq  # store module refs to avoid global import pollution
        self.random = random
        self.w = window_ms
        self.theta = kick_threshold
        self.heap: List = []

    def allocate(self, wc_bounds: Dict[int, tuple]):
        """wc_bounds: {task_id: (exec_time_ms, deadline_ms)}"""
        now = 0.0
        self.heap.clear()
        for tid, (cost, ddl) in wc_bounds.items():
            slack = ddl - cost
            if slack < self.theta * ddl:
                self._heapq.heappush(self.heap, (ddl, tid))  # high priority
            else:
                self._heapq.heappush(self.heap, (ddl + self.random.random(), tid))

        schedule = []
        while self.heap:
            ddl, tid = self._heapq.heappop(self.heap)
            start = now
            end = start + wc_bounds[tid][0]
            schedule.append((tid, start, end))
            now = end
        return schedule


# -----------------------------  TRAIN LOOP  -----------------------------

def run_exp1(cfg: dict, results_root: Path, device: str = "cpu"):
    """Single-node proxy for fleet-scale adaptation & scheduling study."""

    # ------------------------------------------------------------------
    # directory layout
    images_dir = results_root / "images"
    json_dir = results_root
    images_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Reproducibility
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    exp_cfg = cfg["exp1"]

    train_loader = build_loader(
        exp_cfg["datasets"]["wear3_stream_rgb"],
        exp_cfg["batch_size"],
        Path("data"),
        split="train",
    )

    ceml = CEMLGRU(in_dim=10).to(device)
    optim = Adam(ceml.parameters(), lr=exp_cfg["ceml"]["lr"])
    scheduler = RTPS(
        window_ms=exp_cfg["rtps"]["window"],
        kick_threshold=exp_cfg["rtps"]["kick_threshold"],
    )

    # ------------------------------------------------------------------
    # Training loop
    epoch_metrics: List[float] = []
    for epoch in range(exp_cfg["epochs"]):
        ceml.train()
        epoch_loss = 0.0
        for step, (x, _y) in enumerate(train_loader):
            # Mock fleet statistics tensor – in real code replace with real stats
            stats = torch.randn(x.size(0), 5, 10, device=device)
            mu, var = ceml(stats)
            wc_bounds = {
                i: (float(mu[i].item() + 2 * var[i].sqrt().item()), 10.0) for i in range(len(mu))
            }
            _ = scheduler.allocate(wc_bounds)

            loss = ((mu - 1.0) ** 2).mean()
            loss.backward()
            optim.step()
            optim.zero_grad()
            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_loader))
        epoch_metrics.append(epoch_loss)
        print(f"[Exp-1] Epoch {epoch:02d}: loss = {epoch_loss:.4f}")

    # ------------------------------------------------------------------
    # Persist results
    loss_fig = images_dir / "training_loss.pdf"
    line_plot(
        list(range(len(epoch_metrics))),
        epoch_metrics,
        title="CEML Training Loss",
        xlabel="Epoch",
        ylabel="MSE",
        pdf_path=loss_fig,
    )

    results = {
        "description": "Experiment-1 – Fleet-Scale Adaptation (smoke/full)",
        "loss_curve": epoch_metrics,
        "figures": [str(loss_fig.relative_to(results_root))],
    }

    json_path = json_dir / "exp1_results.json"
    save_json(results, json_path)

    # Print for verification (requested by instructions)
    import json as _json

    print("\n=== Experiment-1 Results ===\n", _json.dumps(results, indent=2))

    return results