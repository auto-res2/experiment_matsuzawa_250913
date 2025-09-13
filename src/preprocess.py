# src/preprocess.py
"""Light-weight dataset preparation helpers."""

import json
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
#                     SIMPLE CAPTIONS DATASET (COCO)
# ------------------------------------------------------------------

class COCOCaptionsDataset(Dataset):
    def __init__(self, root: Path, max_samples: int = 300):
        self.root = root; self.max = max_samples; self.captions = []
        self.dir = root / "coco"; self.dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        p = self.dir / "captions_val2014.json"
        if not p.exists():
            try:
                u = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
                z = self.dir / "ann.zip"; logger.info("Downloading COCO captions …")
                urllib.request.urlretrieve(u, z)
                with zipfile.ZipFile(z) as zz:
                    zz.extract("annotations/captions_val2014.json", self.dir)
                z.unlink(); p = self.dir / "annotations" / "captions_val2014.json"
            except Exception as e:
                logger.warning(f"COCO download failed: {e}")
        if p.exists():
            data = json.load(open(p))
            self.captions = [a["caption"] for a in data["annotations"][: self.max]]
        if not self.captions:
            self.captions = [f"synthetic caption {i}" for i in range(self.max)]
        logger.info(f"Loaded {len(self.captions)} captions")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return {"caption": self.captions[idx]}


# ------------------------------------------------------------------
#                  DATASET COLLECTION + DATALOADERS
# ------------------------------------------------------------------

def prepare_datasets(cfg: Dict):
    root = Path("data"); root.mkdir(exist_ok=True)
    ds = {"coco": COCOCaptionsDataset(root, cfg.get("max_coco_samples", 300))}
    return ds


def create_dataloaders(datasets: Dict[str, Dataset], batch_size: int = 4):
    return {k: DataLoader(v, batch_size=batch_size, shuffle=True) for k, v in datasets.items()}
