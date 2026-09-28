"""Train an ESM-2-embedding hemolysis (selectivity) classifier and compare held-out AUROC to
the shipped 11-descriptor model (0.778). Run with the ESM env python (fair-esm + sklearn + CUDA):

    CUDA_VISIBLE_DEVICES=0 <esm-env>/bin/python experiments/esm_selectivity.py

Selectivity is a full scored category (HC50/MIC50) and our least-validated axis, so a stronger
predictor is high-value. ESM-2 embeddings are SOTA for peptide property prediction. If this clearly
beats the current model we integrate it (offline reward + top-100 re-ranking).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
HEMO = REPO / "data" / "hemolysis" / "HemoPI2.fasta"
CACHE = REPO / "experiments" / "cache"


def parse_hemopi2():
    recs = []
    cur = None
    for line in HEMO.read_text().splitlines():
        if line.startswith(">"):
            cur = line[1:]
        elif line.strip():
            recs.append((cur, line.strip()))
    tok = lambda h: h.split("_")[1]  # pm/nm/pv/nv
    train = [(s, 1 if tok(h) == "pm" else 0) for h, s in recs if tok(h) in ("pm", "nm")]
    val = [(s, 1 if tok(h) == "pv" else 0) for h, s in recs if tok(h) in ("pv", "nv")]
    return train, val


@torch.no_grad()
def esm_embed(seqs, model, alphabet, layer, device, batch=64):
    bc = alphabet.get_batch_converter()
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        _, _, toks = bc([(str(j), s) for j, s in enumerate(chunk)])
        toks = toks.to(device)
        rep = model(toks, repr_layers=[layer])["representations"][layer]
        # mean-pool over real residues (exclude BOS at 0 and padding/EOS)
        for k, s in enumerate(chunk):
            out.append(rep[k, 1:len(s) + 1].mean(0).float().cpu().numpy())
    return np.array(out)


def main():
    import esm
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = parse_hemopi2()
    print(f"HemoPI-2: train={len(train)} val={len(val)} "
          f"(train pos {sum(y for _,y in train)}, val pos {sum(y for _,y in val)})")

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    layer = 30
    model = model.eval().to(device)
    Xtr = esm_embed([s for s, _ in train], model, alphabet, layer, device)
    Xva = esm_embed([s for s, _ in val], model, alphabet, layer, device)
    ytr = np.array([y for _, y in train]); yva = np.array([y for _, y in val])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "esm_hemo.npz", Xtr=Xtr, Xva=Xva, ytr=ytr, yva=yva)

    # standardize + logistic regression (strong, low-variance baseline on embeddings)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit((Xtr - mu) / sd, ytr)
    p = clf.predict_proba((Xva - mu) / sd)[:, 1]
    auroc = roc_auc_score(yva, p)
    print(f"ESM-2 (150M) + logistic regression: held-out AUROC = {auroc:.3f}  (shipped model: 0.778)")
    # also a small sklearn MLP
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=0)
    mlp.fit((Xtr - mu) / sd, ytr)
    p2 = mlp.predict_proba((Xva - mu) / sd)[:, 1]
    print(f"ESM-2 (150M) + MLP(128):            held-out AUROC = {roc_auc_score(yva, p2):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
