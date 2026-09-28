"""Selectivity (hemolysis) scorer built on ESM Cambrian 600M embeddings.

The latest SOTA protein language model (ESMC-600M, MIT, loaded via HuggingFace transformers) with
a small trained MLP head -- a large upgrade over the 11 hand-crafted physicochemical descriptors of
:class:`~amp_challenge_2027.hemolysis.HemolysisScorer` (held-out AUROC ~0.89 vs 0.78). Used to
re-rank the most-active top-100 candidates for the Optimal Selectivity category. Inference is
deterministic (``.eval()``, dropout off) on GPU or CPU, so the submission stays byte-reproducible.

Loading ESMC downloads the weights from HuggingFace on first use; if transformers or the weights are
unavailable the caller falls back to the physicochemical hemolysis model, so a valid submission is
always produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import resolve_repo_path

DEFAULT_SELECTIVITY_CHECKPOINT = "checkpoint/selectivity_esmc.pt"


class EsmcSelectivityScorer:
    """P(hemolytic) from ESMC-600M embeddings + a trained MLP head. Higher = more hemolytic."""

    def __init__(self, checkpoint: str | Path = DEFAULT_SELECTIVITY_CHECKPOINT,
                 *, device: str | None = None, batch_size: int = 32) -> None:
        import torch
        import torch.nn as nn
        from transformers import AutoModel

        ckpt = torch.load(resolve_repo_path(checkpoint), map_location="cpu", weights_only=False)
        self._torch = torch
        self._mu = torch.tensor(ckpt["mu"])
        self._sd = torch.tensor(ckpt["sd"])
        self._batch = batch_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # ESM++ (ESMC-600M) backbone with its own bundled tokenizer (trust_remote_code).
        hf = ckpt["hf_model"]
        self._esm = AutoModel.from_pretrained(hf, trust_remote_code=True).eval().to(self.device)
        self._tok = self._esm.tokenizer
        d = self._mu.shape[0]
        self._head = nn.Sequential(
            nn.Linear(d, ckpt["hidden"]), nn.ReLU(), nn.Dropout(0.3), nn.Linear(ckpt["hidden"], 1)
        )
        self._head.load_state_dict(ckpt["state_dict"])
        self._head.eval().to(self.device)
        self._mu = self._mu.to(self.device)
        self._sd = self._sd.to(self.device)

    def predict_proba(self, sequences: list[str]) -> np.ndarray:
        """P(hemolytic) per sequence (deterministic; eval mode)."""
        torch = self._torch
        if not sequences:
            return np.empty(0, dtype=float)
        probs: list[float] = []
        with torch.no_grad():
            for start in range(0, len(sequences), self._batch):
                chunk = sequences[start:start + self._batch]
                enc = self._tok(chunk, return_tensors="pt", padding=True)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                h = self._esm(**enc).last_hidden_state  # (B, T, d)
                embs = torch.stack([
                    h[k, 1:1 + len(s)].float().mean(0) for k, s in enumerate(chunk)
                ])
                logits = self._head((embs - self._mu) / self._sd).squeeze(1)
                probs.extend(torch.sigmoid(logits).cpu().tolist())
        return np.asarray(probs, dtype=float)
