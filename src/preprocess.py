"""Lightweight dataset preparation utilities (downloads stubbed if offline)."""

from __future__ import annotations

import json
import logging
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# Simple COCO caption subset (fallback to synthetic captions if offline)
# ----------------------------------------------------------------------------


class COCOCaptionsDataset(Dataset):
    def __init__(self, root: Path, max_samples: int = 300):
        self.captions = []
        ann_dir = root / "coco"
        ann_dir.mkdir(parents=True, exist_ok=True)
        j_path = ann_dir / "captions_val2014.json"
        if not j_path.exists():
            try:
                logger.info("Downloading COCO captions …")
                url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
                tmp = ann_dir / "ann.zip"
                urllib.request.urlretrieve(url, tmp)
                with zipfile.ZipFile(tmp) as zf:
                    zf.extract("annotations/captions_val2014.json", ann_dir)
                tmp.unlink()
            except Exception as e:  # pragma: no cover
                logger.warning("COCO download failed (%s); generating synthetic captions", e)
                self.captions = [f"A synthetic caption {i}" for i in range(max_samples)]
        if not self.captions:
            with open(j_path) as fp:
                data = json.load(fp)
            for ann in data["annotations"][: max_samples]:
                self.captions.append(ann["caption"])
        self.captions = self.captions[:max_samples]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return {"caption": self.captions[idx]}


# ----------------------------------------------------------------------------
# Public helpers for main
# ----------------------------------------------------------------------------


def prepare_datasets(_cfg: Dict):  # noqa: D401
    data_root = Path("data"); data_root.mkdir(exist_ok=True)
    coco = COCOCaptionsDataset(data_root)
    return {"coco": coco}


def create_dataloaders(ds: Dict[str, Dataset], bs: int = 4):  # noqa: D401
    return {k: DataLoader(v, batch_size=bs, shuffle=True, num_workers=0) for k, v in ds.items()}
