"""
Entry-point orchestrating smoke-test & full-experiment modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from .preprocess import FederatedGraphDataset
from .train import (
    FedC3PO,
    FedC3POConfig,
    FederatedClient,
    FederatedServer,
    plot_training_curves,
    save_training_results,
    train_federated,
)
from .evaluate import evaluate_model, save_evaluation_results


def _load_yaml(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def _run_pipeline(cfg: dict):
    ds_name = cfg["experiment_1"]["datasets"][0]
    fed_cfg = FedC3POConfig(
        num_clients=cfg["experiment_1"]["num_clients"],
        rounds=cfg["experiment_1"]["rounds"],
        local_epochs=cfg["experiment_1"]["local_epochs"],
    )
    dataset = FederatedGraphDataset(ds_name, fed_cfg.num_clients)
    n_feat = dataset.data.x.size(1)
    n_cls = int(dataset.data.y.max().item() + 1)
    global_model = FedC3PO(fed_cfg, n_feat, n_cls)
    server = FederatedServer(global_model, fed_cfg)
    clients = [
        FederatedClient(cid, FedC3PO(fed_cfg, n_feat, n_cls), dataset.get_client_loader(cid), fed_cfg)
        for cid in range(fed_cfg.num_clients)
    ]
    test_loader = dataset.get_test_loader()
    metrics = train_federated(fed_cfg, clients, server, test_loader)
    save_training_results(metrics, ".research/iteration2/exp_smoke_train.json")
    plot_training_curves(metrics, ".research/iteration2/images/smoke_curves.png")
    eval_metrics, _, _ = evaluate_model(server.model, test_loader, fed_cfg)
    save_evaluation_results(eval_metrics, ".research/iteration2/exp_smoke_eval.json")
    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description="FedC3PO launcher")
    parser.add_argument("--smoke-test", action="store_true", help="run quick smoke-test then exit or proceed")
    parser.add_argument("--full-experiment", action="store_true", help="run full experiment (will first run smoke-test)")
    args = parser.parse_args()

    Path(".research/iteration2/images").mkdir(parents=True, exist_ok=True)
    Path(".research/iteration2").mkdir(parents=True, exist_ok=True)

    if not (args.smoke_test or args.full_experiment):
        print("Specify --smoke-test or --full-experiment")
        sys.exit(1)

    # Always run smoke test first (requirement)
    print("===== SMOKE TEST =====")
    smoke_cfg = _load_yaml("config/smoke_test.yaml")
    smoke_metrics = _run_pipeline(smoke_cfg)
    if args.smoke_test:
        print("Smoke-test completed ✓ – exiting.")
        return

    # If full experiment requested, only continue when smoke-test passes minimal condition
    if smoke_metrics.get("accuracy", 0) < 0.30:
        print("Smoke-test accuracy too low – aborting full experiment.")
        sys.exit(1)

    print("\n===== FULL EXPERIMENT =====")
    full_cfg = _load_yaml("config/full_experiment.yaml")
    _run_pipeline(full_cfg)
    print("Full experiment completed ✓")


if __name__ == "__main__":
    main()
