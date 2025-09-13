import os
from pathlib import Path
from typing import Dict

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .evaluate import download  # reuse util, avoids duplication


# ---------------------------  DATASETS & LOADERS  ---------------------------
class Wear3RGBSmall(torch.utils.data.Dataset):
    """CIFAR-10 proxy for Wear3 RGB stream (tiny, quick)."""

    def __init__(self, root: Path, split: str = "train"):
        root.mkdir(parents=True, exist_ok=True)
        train = split == "train"
        self.ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = T.Compose(
            [T.Resize(160), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, lbl = self.ds[idx]
        return self.transform(img), lbl


# NOTE: In a real full experiment additional dataset wrappers (SpikeWear-DVS,
# PhysioEvent-ECG, etc.) would be implemented here.  For brevity they raise.

def build_loader(cfg_ds: Dict, batch: int, root: Path, split: str):
    """Return a PyTorch DataLoader according to *cfg_ds* spec."""

    name = cfg_ds.get("hf_name", "")

    if name.lower() == "cifar10":
        dataset = Wear3RGBSmall(root, split)
    else:
        # Fallback: try download from URL if provided, then raise
        if "url" in cfg_ds:
            fn = Path(cfg_ds["url"]).name
            download(cfg_ds["url"], root / fn, cfg_ds.get("checksum"))
        raise FileNotFoundError(
            f"No loader implemented for dataset spec: {cfg_ds}. Extend preprocess.py."
        )

    num_workers = min(4, os.cpu_count() or 1)
    return DataLoader(dataset, batch_size=batch, shuffle=(split == "train"), num_workers=num_workers)
