from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision.transforms as T
from PIL import Image
from datasets import load_dataset

__all__ = ["DataPreprocessor"]


class _SplitCIFAR100(Dataset):
    """CIFAR-100 → 25 tasks ×4 classes  (+ simple OOD distractor)"""

    def __init__(self, train: bool, transform, num_tasks: int = 25, seed: int = 42):
        self.data = load_dataset("uoft-cs/cifar100", split="train" if train else "test")
        self.transform = transform
        self.num_tasks = num_tasks
        rng = np.random.RandomState(seed)
        classes = np.arange(100)
        rng.shuffle(classes)
        self.task_cls = [classes[i * 4 : (i + 1) * 4].tolist() for i in range(num_tasks)]
        self.cls2task = {c: t for t, cls in enumerate(self.task_cls) for c in cls}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samp = self.data[idx]
        img = samp["img"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = self.transform(img)
        return img, samp["fine_label"]

    def task_indices(self, tid: int):
        return [i for i, samp in enumerate(self.data) if samp["fine_label"] in self.task_cls[tid]]

    def loader(self, tid: int, bs: int = 32, shuffle: bool = True):
        return DataLoader(Subset(self, self.task_indices(tid)), batch_size=bs, shuffle=shuffle)


class DataPreprocessor:
    """Wrapper giving get_task_loader API used by training code"""

    def __init__(self, dataset_name: str = "vision", root: str = "data"):
        self.dataset_name = dataset_name
        root = Path(root)
        root.mkdir(exist_ok=True)
        if dataset_name == "vision":
            tr = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            te = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.train_ds = _SplitCIFAR100(True, tr)
            self.test_ds = _SplitCIFAR100(False, te)
        else:
            raise NotImplementedError("Smoke-test only vision dataset is supported here.")

    # api used by trainer
    def get_task_loader(self, tid: int, batch_size: int = 32, train: bool = True):
        return (self.train_ds if train else self.test_ds).loader(tid, batch_size)

# Make discoverable as top-level module when imported via "preprocess_py" --------
import sys as _sys
_sys.modules.setdefault("preprocess_py", _sys.modules[__name__])
