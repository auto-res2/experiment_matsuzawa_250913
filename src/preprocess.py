"""src/preprocess.py
Dataset modules and auxiliary helpers.
"""
from __future__ import annotations

import pathlib
from typing import Dict, Any, List

from torch.utils.data import ConcatDataset, DataLoader
import torch

###############################################################################
#  Text  (HF datasets)
###############################################################################
from datasets import load_dataset
from transformers import AutoTokenizer


class TextDataModule:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"], use_fast=True)

    # ------------------------------------------------------------------
    def _hf(self, name: str, subset: str | None, split: str, streaming: bool):
        return load_dataset(name, subset, split=split, streaming=streaming)

    # ------------------------------------------------------------------
    def prepare(self):
        parts: List[Any] = []
        for ds in self.cfg["datasets"]:
            name = ds["hf_name"]
            subset = ds.get("subset")
            split = ds.get("split", "train")
            streaming = bool(ds.get("streaming", False))
            parts.append(self._hf(name, subset, split, streaming))
        self.ds = ConcatDataset(parts)

    # ------------------------------------------------------------------
    def _collate(self, batch):
        texts = [x["text"] for x in batch]
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg["max_length"],
            return_tensors="pt",
        )
        return {k: v for k, v in tokens.items() if k in ("input_ids", "attention_mask")}

    # ------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            collate_fn=self._collate,
            num_workers=4,
            pin_memory=True,
        )

###############################################################################
#  BTCV  3-D CT  (MONAI)
###############################################################################
try:
    import monai
    from monai.transforms import (
        Compose,
        LoadImaged,
        Spacingd,
        ScaleIntensityRanged,
        RandFlipd,
        RandRotate90d,
        ToTensord,
    )
    from monai.data import CacheDataset
except ImportError as e:  # pragma: no cover
    raise RuntimeError("monai is required for BTCV preprocessing") from e


class BTCVDataModule:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.data_dir = pathlib.Path(cfg["root"]).expanduser().resolve()
        if not self.data_dir.exists():
            raise RuntimeError(
                "BTCV dataset directory missing – aborting per NO-FALLBACK rule. "
                f"Expected at {self.data_dir}"
            )
        self.train_files = sorted((self.data_dir / "train").glob("*.nii.gz"))

    # ------------------------------------------------------------------
    def prepare(self):
        tr = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=250, b_min=-1, b_max=1, clip=True),
                RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
                ToTensord(keys=["image", "label"]),
            ]
        )
        data = [
            {"image": str(f), "label": str(f).replace("image", "label")} for f in self.train_files
        ]
        self.ds = CacheDataset(data=data, transform=tr, cache_rate=0.1)

    # ------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
