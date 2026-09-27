"""Char-level autoregressive Transformer over the 20-AA alphabet.

The single source of truth for the generator architecture, used both to train on the
H100s (``training/``) and to sample inside ``uv run generate``. Inference is deterministic
and runs on GPU when available (the challenge validator has one — the official AMP-Diffusion
baseline requires it) and falls back to CPU. ``torch`` is therefore a runtime dependency;
``build_model`` degrades to a non-neural baseline if it or the checkpoint is missing, so the
submission never hard-fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary: 3 special tokens then the 20 standard amino acids, in the canonical order
# used everywhere else in the package (constraints.ALPHABET).
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD, BOS, EOS = 0, 1, 2
STOI = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS, **{a: i + 3 for i, a in enumerate(ALPHABET)}}
ITOS = {i: t for t, i in STOI.items()}
VOCAB_SIZE = len(STOI)  # 23
MAX_RESIDUES = 50
MAX_LEN = MAX_RESIDUES + 2  # BOS + 50 residues + EOS


def encode(seq: str) -> list[int]:
    """[BOS] + residues + [EOS] as token ids."""
    return [BOS] + [STOI[a] for a in seq] + [EOS]


def decode(ids: list[int]) -> str:
    """Token ids back to a residue string, stopping at EOS and dropping specials."""
    out = []
    for i in ids:
        if i == EOS:
            break
        if i in (PAD, BOS):
            continue
        out.append(ITOS[i])
    return "".join(out)


@dataclass
class LMConfig:
    vocab_size: int = VOCAB_SIZE
    max_len: int = MAX_LEN
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1

    def as_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size, "max_len": self.max_len, "d_model": self.d_model,
            "n_layers": self.n_layers, "n_heads": self.n_heads, "d_ff": self.d_ff,
            "dropout": self.dropout,
        }


class Block(nn.Module):
    """Pre-LN decoder block: causal multi-head self-attention + MLP."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class PeptideLM(nn.Module):
    """Decoder-only Transformer LM. Learned token + positional embeddings, tied output head."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, T] -> logits [B, T, vocab]."""
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.drop(self.tok(tokens) + self.pos(pos))
        # Boolean masks throughout (True = disallowed) so key_padding and attn masks match.
        causal = torch.ones((T, T), dtype=torch.bool, device=tokens.device).triu(diagonal=1)
        key_padding = tokens == PAD
        for block in self.blocks:
            x = block(x, causal, key_padding)
        return self.head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def sample(
    model: PeptideLM,
    n: int,
    *,
    device: torch.device,
    temperature: float = 1.0,
    top_p: float = 1.0,
    min_residues: int = 8,
    max_residues: int = MAX_RESIDUES,
    generator: torch.Generator | None = None,
    batch_size: int = 4096,
) -> list[str]:
    """Autoregressively sample `n` sequences. Used for in-training novelty checks only;
    the shipped sampler is the numpy port. Sequences outside the length window are dropped
    (the caller oversamples)."""
    model.eval()
    cfg = model.cfg
    out: list[str] = []
    while len(out) < n:
        b = min(batch_size, n - len(out) + batch_size // 4)
        tokens = torch.full((b, 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(cfg.max_len - 1):
            logits = model(tokens)[:, -1, :] / max(temperature, 1e-6)
            logits[:, PAD] = float("-inf")
            logits[:, BOS] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                probs = _nucleus(probs, top_p)
            nxt = torch.multinomial(probs, 1, generator=generator)
            nxt[finished] = PAD
            tokens = torch.cat([tokens, nxt], dim=1)
            finished = finished | (nxt.squeeze(1) == EOS)
            if bool(finished.all()):
                break
        for row in tokens.tolist():
            seq = decode(row)
            if min_residues <= len(seq) <= max_residues:
                out.append(seq)
    return out[:n]


def _nucleus(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return probs / probs.sum(dim=-1, keepdim=True)


def load_generator(path: str | Path, device: torch.device) -> PeptideLM:
    """Load a trained checkpoint (``{state_dict, config, ...}``) into an eval-mode model."""
    ckpt = torch.load(Path(path), map_location=device)
    model = PeptideLM(LMConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def sequence_nll(
    model: PeptideLM,
    sequences: list[str],
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> list[float]:
    """Mean per-residue negative log-likelihood under the model, one per sequence.

    Lower means the model finds the sequence more AMP-like. Used as the interim ranking
    signal for the top 100 until the APEX activity oracle (feat-013) replaces it. Padded
    positions are excluded from the average, so length does not bias the score.
    """
    model.eval()
    scores: list[float] = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        maxlen = max(len(encode(s)) for s in chunk)
        batch = torch.full((len(chunk), maxlen), PAD, dtype=torch.long, device=device)
        for i, s in enumerate(chunk):
            ids = encode(s)
            batch[i, : len(ids)] = torch.tensor(ids, device=device)
        logits = model(batch[:, :-1])
        target = batch[:, 1:]
        logp = F.log_softmax(logits.float(), dim=-1)
        tok_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        mask = target != PAD
        nll = -(tok_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        scores.extend(nll.tolist())
    return scores
