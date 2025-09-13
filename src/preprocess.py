import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset

__all__ = ["DataPreprocessor"]


class CIFAR100Task(Dataset):
    """Split CIFAR-100 into sequential tasks (default 4 classes per task)."""

    def __init__(self, split: str, task: int, cls_per_task: int = 4):
        self.ds = load_dataset("uoft-cs/cifar100", split=split)
        self.tr = (
            T.Compose([
                T.RandomCrop(32, 4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            if split == "train"
            else T.Compose([
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        )
        st = task * cls_per_task
        self.classes = list(range(st, min(st + cls_per_task, 100)))
        self.idxs = [i for i in range(len(self.ds)) if self.ds[i]["fine_label"] in self.classes]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        s = self.ds[self.idxs[i]]
        img = self.tr(s["img"])
        y = self.classes.index(s["fine_label"])
        return img, y


class DataPreprocessor:
    """Factory that returns task-wise data loaders for a given modality."""

    def __init__(self, modality: str = "vision"):
        self.mod = modality.lower()
        if self.mod != "vision":
            raise ValueError("Only vision modality implemented in this refactor.")

    def get_task_loaders(self, split: str, n_tasks: int, batch: int = 32) -> List[DataLoader]:
        loaders = []
        for t in range(n_tasks):
            ds = CIFAR100Task(split, t)
            loaders.append(
                DataLoader(ds, batch_size=batch, shuffle=(split == "train"), num_workers=2, pin_memory=True)
            )
        return loaders
