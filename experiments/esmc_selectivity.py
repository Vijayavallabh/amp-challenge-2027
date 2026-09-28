"""Benchmark ESMC-600M (latest SOTA, MIT, local) on the HemoPI-2 selectivity task and check
determinism. Run with the ESMC env python (EvolutionaryScale `esm` SDK + sklearn + CUDA).

Compares held-out AUROC to ESM-2 150M (0.883) and the shipped physchem model (0.778), and
verifies GPU (bf16) + CPU embeddings are byte-reproducible (needed for the shipped two-run check).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
HEMO = REPO / "data" / "hemolysis" / "HemoPI2.fasta"
CACHE = REPO / "experiments" / "cache"


def parse():
    recs, cur = [], None
    for line in HEMO.read_text().splitlines():
        if line.startswith(">"):
            cur = line[1:]
        elif line.strip():
            recs.append((cur, line.strip()))
    tok = lambda h: h.split("_")[1]
    tr = [(s, 1 if tok(h) == "pm" else 0) for h, s in recs if tok(h) in ("pm", "nm")]
    va = [(s, 1 if tok(h) == "pv" else 0) for h, s in recs if tok(h) in ("pv", "nv")]
    return tr, va


def main():
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ESMC.from_pretrained("esmc_600m").to(dev).eval()
    cfg = LogitsConfig(sequence=True, return_embeddings=True)

    @torch.no_grad()
    def emb(s):
        t = m.encode(ESMProtein(sequence=s))
        out = m.logits(t, cfg)
        return out.embeddings[0, 1:1 + len(s)].mean(0).float().cpu().numpy()

    def emb_all(seqs):
        return np.array([emb(s) for s in seqs], dtype=np.float32)

    tr, va = parse()
    print(f"HemoPI-2: train={len(tr)} val={len(va)}")
    t0 = time.time()
    Xtr = emb_all([s for s, _ in tr]); Xva = emb_all([s for s, _ in va])
    ytr = np.array([y for _, y in tr]); yva = np.array([y for _, y in va])
    print(f"ESMC-600M embed dim={Xtr.shape[1]} in {time.time()-t0:.0f}s "
          f"({(len(tr)+len(va))/(time.time()-t0):.0f}/s)")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "esmc_hemo.npz", Xtr=Xtr, Xva=Xva, ytr=ytr, yva=yva)

    # determinism check
    a = emb("WLRWALKRIYRWNWK"); b = emb("WLRWALKRIYRWNWK")
    print(f"ESMC embed deterministic (2 runs, {dev} bf16): {np.array_equal(a,b)} max|diff|={np.abs(a-b).max():.2e}")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    lr = LogisticRegression(max_iter=3000).fit((Xtr - mu) / sd, ytr)
    au_lr = roc_auc_score(yva, lr.predict_proba((Xva - mu) / sd)[:, 1])
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=0).fit((Xtr - mu) / sd, ytr)
    au_mlp = roc_auc_score(yva, mlp.predict_proba((Xva - mu) / sd)[:, 1])
    print(f"ESMC-600M + LR:      held-out AUROC = {au_lr:.3f}")
    print(f"ESMC-600M + MLP(128):held-out AUROC = {au_mlp:.3f}   (ESM-2 150M: 0.883, physchem: 0.778)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
