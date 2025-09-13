import hashlib
import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, List

import matplotlib
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

matplotlib.use("Agg")  # headless backend


# -----------------------------  FILE UTILITIES  -----------------------------
class DownloadError(RuntimeError):
    """Raised when dataset / asset download fails or checksum mismatches."""


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path, checksum: str | None = None) -> Path:
    """Download *url* to *dst* (if necessary) and optionally verify sha256 checksum.
    Archives (.tar, .zip) are extracted automatically next to *dst*.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Short-circuit if already present & checksum (if given) matches
    if dst.exists() and (checksum is None or sha256sum(dst) == checksum.split(":")[-1]):
        return dst

    try:
        r = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as e:
        raise DownloadError(f"Failed to GET {url} – {e}") from e

    if r.status_code != 200:
        raise DownloadError(f"HTTP {r.status_code}: {url}")

    total = int(r.headers.get("content-length", 0))
    with open(dst, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"DL {dst.name}") as p:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            p.update(len(chunk))

    if checksum is not None and sha256sum(dst) != checksum.split(":")[-1]:
        raise DownloadError(f"Checksum mismatch for {dst}")

    # Auto-extract archives
    try:
        if tarfile.is_tarfile(dst):
            with tarfile.open(dst) as tar:
                tar.extractall(dst.parent)
        elif zipfile.is_zipfile(dst):
            with zipfile.ZipFile(dst) as z:
                z.extractall(dst.parent)
    except Exception as e:
        # Non-fatal – proceed with downloaded file but warn user
        print(f"[WARN] Failed to auto-extract {dst}: {e}")

    return dst


# ------------------------------  PLOTTING  ----------------------------------

def line_plot(xs: Iterable[Any], ys: Iterable[float], title: str, xlabel: str, ylabel: str, pdf_path: Path):
    plt.figure()
    plt.plot(list(xs), list(ys), marker="o", label=title)
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.2f}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


# ------------------------------  JSON IO  -----------------------------------

def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
