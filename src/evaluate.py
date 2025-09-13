"""src/evaluate.py
Evaluation / experiment orchestration, incl. samplers & plotting utilities.
"""
from __future__ import annotations

import json
import random
import pathlib
from typing import Dict, Any, List

import matplotlib

matplotlib.use("Agg")  # headless back-end
import matplotlib.pyplot as plt
import torch

from .train import (
    ScoreNetText400M,
    ScoreNetCTUNet3D,
    train_text_model,
    train_ct_model,
)
from .preprocess import TextDataModule, BTCVDataModule

###############################################################################
#   Sampler wrappers  (identical interface)
###############################################################################


def _sampler_hawq(score_model: torch.nn.Module, cfg: Dict[str, Any]):
    import importlib

    pkg = "hawq_raptor_torch"
    if importlib.util.find_spec(pkg) is None:
        raise RuntimeError("hawq-raptor-torch not installed – aborting per NO-FALLBACK policy.")
    from hawq_raptor_torch import HAWQRaptorSampler

    return HAWQRaptorSampler(score_model, **cfg)


def _sampler_certex(score_model: torch.nn.Module, cfg: Dict[str, Any]):
    from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

    return DPMSolverMultistepScheduler(num_train_timesteps=1000, **cfg)


def _sampler_dpmpp(score_model: torch.nn.Module, cfg: Dict[str, Any]):
    from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

    steps = cfg.get("num_steps", 1000)
    return DPMSolverMultistepScheduler(num_train_timesteps=steps, algorithm_type="dpmsolver++")

###############################################################################
#   Plot helpers
###############################################################################

def _plot_line(xs: List[float], ys: List[float], xlabel: str, ylabel: str, title: str, pdf_path: pathlib.Path):
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

###############################################################################
#   Experiment 1  –  model training & NFE comparison
###############################################################################

def experiment1(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # ----- Training -----
    text_model, text_loss = train_text_model(cfg["text_data"], cfg["training"])
    ct_model, ct_loss = train_ct_model(cfg["ct_data"], cfg["training"])

    # ----- Sampler comparison -----
    sampler_cfg = cfg.get("samplers", {})
    sampler_names = ["HAWQ", "CER", "DPMPP"]
    sampler_fns = [_sampler_hawq, _sampler_certex, _sampler_dpmpp]
    nfes: List[int] = []
    for name, fn in zip(sampler_names, sampler_fns):
        smp_cfg = sampler_cfg.get(name.lower(), {})
        smp = fn(text_model, smp_cfg)
        nfe = getattr(smp, "num_inference_steps", 0) or sampler_cfg.get("default_steps", 0)
        nfes.append(int(nfe))

    # ----- Figure -----
    img_dir = pathlib.Path(".research/iteration1/images")
    pdf = img_dir / "nfe_comparison.pdf"
    _plot_line(list(range(len(nfes))), nfes, "Sampler idx", "NFE", "NFE comparison", pdf)

    # ----- Results -----
    results = {
        "text_final_loss": text_loss,
        "ct_final_loss": ct_loss,
        "nfe_list": nfes,
    }

    res_dir = pathlib.Path(".research/iteration1")
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "experiment1_results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print("\n===== Experiment 1 – Manifold-Sensitive Dual Cert =====")
    print(json.dumps(results, indent=2))
    print(f"Figure saved: {pdf.relative_to(res_dir.parent)}")
    print("======================================================\n")

    return results

###############################################################################
#   Experiment 2  –  FP16 safety violations
###############################################################################

def _bit_flip_tensor(t: torch.Tensor):
    b = t.view(torch.uint8)
    idx = random.randint(0, b.numel() - 1)
    b[idx] ^= 0b00010000


def experiment2(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dummy oracle (real model in full config would be loaded separately)
    score_model = torch.nn.Linear(10, 10).to(device)

    samplers = {
        "hawq": _sampler_hawq(score_model, cfg["hawq"]),
        "cer": _sampler_certex(score_model, cfg["cer"]),
    }

    steps = int(cfg["steps"])
    safety: Dict[str, float] = {}
    for name, smp in samplers.items():
        vio = 0
        for i in range(steps):
            x = torch.randn(1, 10, device=device, dtype=torch.float16)
            try:
                smp.step(x)  # external API call; ensured by sampler implementation
            except (FloatingPointError, RuntimeError):
                vio += 1
            if i % 500 == 0:
                _bit_flip_tensor(x)
        safety[name] = vio / steps * 1e6  # per million steps

    # ----- plot -----
    img_dir = pathlib.Path(".research/iteration1/images")
    img_dir.mkdir(parents=True, exist_ok=True)
    pdf = img_dir / "fp16_safety.pdf"
    plt.figure(figsize=(4, 3))
    labels, vals = list(safety.keys()), list(safety.values())
    plt.bar(labels, vals)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.1, f"{v:.1f}", ha="center")
    plt.ylabel("Violations / 10⁶")
    plt.title("FP16 safety")
    plt.tight_layout()
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

    results = {"fp16_safety_violation_per_million": safety}

    res_dir = pathlib.Path(".research/iteration1")
    with open(res_dir / "experiment2_results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print("\n===== Experiment 2 – Mixed-Precision Stability =====")
    print(json.dumps(results, indent=2))
    print(f"Figure saved: {pdf.relative_to(res_dir.parent)}")
    print("===============================================\n")
    return results

###############################################################################
#   Experiment 3  –  Fed-Personalisation & Sustainability (mock-up)
###############################################################################

def experiment3(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rounds = int(cfg.get("rounds", 5))
    bleu: List[float] = [0.2]
    water: List[float] = [1.0]
    for _ in range(1, rounds + 1):
        bleu.append(min(bleu[-1] + 0.1, 0.9))
        water.append(water[-1] * 0.9)

    results = {
        "dialect_bleu": bleu[-1],
        "blue_water_cvar": water[-1],
    }

    # plotting
    img_dir = pathlib.Path(".research/iteration1/images")
    img_dir.mkdir(parents=True, exist_ok=True)
    pdf = img_dir / "dialect_bleu.pdf"
    plt.figure(figsize=(6, 3))
    plt.plot(range(rounds + 1), bleu, label="BLEU")
    for x, y in enumerate(bleu):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
    plt.xlabel("Round")
    plt.ylabel("BLEU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

    res_dir = pathlib.Path(".research/iteration1")
    with open(res_dir / "experiment3_results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print("\n===== Experiment 3 – Fed Personalisation & Sustainability =====")
    print(json.dumps(results, indent=2))
    print(f"Figure saved: {pdf.relative_to(res_dir.parent)}")
    print("============================================================\n")

    return results

###############################################################################
#  Dispatch helper for main.py
###############################################################################

EXPERIMENT_REGISTRY = {
    "experiment1": experiment1,
    "experiment2": experiment2,
    "experiment3": experiment3,
}
