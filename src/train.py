"""src/train.py
All model definitions and training helpers live here.
"""
from __future__ import annotations

import time
from typing import Dict, Any, Tuple

import torch
from torch import nn
from torch.optim import AdamW

from .preprocess import TextDataModule, BTCVDataModule

###############################################################################
#  Model definitions
###############################################################################

class _GPTBlock(nn.Module):
    """A minimal GPT transformer block (pre-LN)."""

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            n_embd, n_head, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:  # noqa: D401,E501
        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class ScoreNetText400M(nn.Module):
    """24-layer GPT-style decoder (~400 M params) usable as a score-network."""

    def __init__(
        self, vocab_size: int, n_layer: int = 24, n_embd: int = 1024, n_head: int = 16, dropout: float = 0.1
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, 2048, n_embd))
        self.blocks = nn.ModuleList([_GPTBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:  # noqa: E501
        bsz, seq_len = input_ids.size()
        x = self.tok_emb(input_ids) + self.pos_emb[:, :seq_len, :]
        for blk in self.blocks:
            x = blk(x, attn_mask=None)
        x = self.ln_f(x)
        return self.output(x)


class ScoreNetCTUNet3D(nn.Module):
    """3-D UNet++ variant from MONAI for volumetric diffusion."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024),
    ):
        super().__init__()
        try:
            import monai.networks.nets as monai_nets
        except ImportError as e:  # pragma: no cover – dependency missing
            raise RuntimeError("monai is required for 3-D UNet++ network") from e

        self.unet = monai_nets.UNet(
            dimensions=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=(2, 2, 2, 2, 2, 2),
            num_res_units=2,
            norm="GN",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.unet(x)


###############################################################################
#  Training helpers
###############################################################################

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_text_model(text_cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> Tuple[ScoreNetText400M, float]:
    """Returns the trained model + final CE loss."""

    device = _get_device()

    # data-module
    dm = TextDataModule(text_cfg)
    dm.prepare()

    model = ScoreNetText400M(dm.tokenizer.vocab_size).to(device)

    opt = AdamW(model.parameters(), lr=train_cfg["lr_text"])
    loader = dm.train_dataloader()

    max_steps = int(train_cfg["max_steps_text"])
    ce_loss = torch.nn.functional.cross_entropy

    model.train()
    for step, batch in enumerate(loader):
        opt.zero_grad(set_to_none=True)
        inp = batch["input_ids"].to(device)
        logits = model(inp)
        loss = ce_loss(logits.view(-1, logits.size(-1)), inp.view(-1))
        loss.backward()
        opt.step()
        if step >= max_steps:
            break

    return model, float(loss.detach().cpu())


def train_ct_model(ct_cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> Tuple[ScoreNetCTUNet3D, float]:
    device = _get_device()

    dm = BTCVDataModule(ct_cfg)
    dm.prepare()

    model = ScoreNetCTUNet3D().to(device)
    opt = AdamW(model.parameters(), lr=train_cfg["lr_ct"])

    max_steps = int(train_cfg["max_steps_ct"])

    model.train()
    for step, batch in enumerate(dm.train_dataloader()):
        opt.zero_grad(set_to_none=True)
        img = batch["image"].to(device)
        out = model(img)
        loss = ((out - img) ** 2).mean()
        loss.backward()
        opt.step()
        if step >= max_steps:
            break

    return model, float(loss.detach().cpu())