"""Measure whether the stronger ESM-2 selectivity model would improve top-100 selection.

Run with the ESM env python (CUDA). Trains the ESM-2 hemolysis MLP from cached embeddings,
then on a representative pool compares the ESM-2-predicted hemolysis of:
  (a) top-100 by APEX activity alone (no selectivity),
  (b) top-100 by activity - lambda*ESM2_hemolysis,
  (c) the actual shipped top-100 (generate/top.fasta),
so we can see if ESM-2-guided selection finds more-selective actives that the shipped
physchem model missed. Decision input for whether to integrate ESM-2 selectivity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "experiments" / "cache"
GN = [0, 1, 2, 3, 4, 5, 6]; GP = [7, 8, 9, 10]; MDR = [8, 9, 10]; THRESH = 16.0


def soft_success(mic):
    return 1.0 / (1.0 + np.exp(-((np.log10(THRESH) - np.log10(np.clip(mic, 1e-6, None))) / 0.5)))


def activity(mic):
    s = soft_success(mic)
    return s.mean(1) + 0.5 * s[:, GP].mean(1) + 0.5 * s[:, MDR].mean(1)


def breadth(mic):
    return (mic <= THRESH).sum(1)


@torch.no_grad()
def embed(seqs, model, alphabet, device, batch=128):
    bc = alphabet.get_batch_converter()
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        _, _, toks = bc([(str(j), s) for j, s in enumerate(chunk)])
        rep = model(toks.to(device), repr_layers=[30])["representations"][30]
        for k, s in enumerate(chunk):
            out.append(rep[k, 1:len(s) + 1].mean(0).float().cpu().numpy())
    return np.array(out, dtype=np.float32)


def read_fasta(p):
    return [l.strip() for l in Path(p).read_text().splitlines() if l and not l.startswith(">")]


def main():
    import esm
    from sklearn.neural_network import MLPClassifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) train ESM-2 hemolysis MLP from cached embeddings
    d = np.load(CACHE / "esm_hemo.npz")
    mu, sd = d["Xtr"].mean(0), d["Xtr"].std(0) + 1e-8
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=0)
    clf.fit((d["Xtr"] - mu) / sd, d["ytr"])
    def phemo(emb):
        return clf.predict_proba((emb - mu) / sd)[:, 1]

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.eval().to(device)

    # 2) representative pool with APEX scores
    pool = np.load(CACHE / "c3r3_T1.6.npz", allow_pickle=True)
    pseq = list(pool["sequences"]); pmic = pool["mic"].astype(float)
    act = activity(pmic)
    top_act = np.argsort(-act)[:8000]                 # activity candidates
    emb = embed([pseq[i] for i in top_act], model, alphabet, device)
    ph = phemo(emb)
    minmic = pmic[top_act].min(1)
    active = minmic <= THRESH

    def report(idx_local, tag):
        b = breadth(pmic[top_act][idx_local]).mean()
        print(f"  {tag:<34} n={len(idx_local)} breadth={b:.2f}/11 "
              f"ESM2 P(hemo) median={np.median(ph[idx_local]):.3f} mean={ph[idx_local].mean():.3f}")

    a = act[top_act]
    sel_act = np.argsort(-a)[:100]                              # activity only
    sel_esm = np.argsort(-(a - 0.5 * ph))[:100]                 # activity - 0.5*ESM2_hemo
    print(f"pool={len(pseq)}  candidates(top-8000 active)={active.sum()} active")
    report(sel_act, "(a) top-100 activity only")
    report(sel_esm, "(b) top-100 activity-0.5*ESM2hemo")

    # 3) shipped top-100
    tseq = read_fasta(REPO / "generate" / "top.fasta")
    temb = embed(tseq, model, alphabet, device)
    tph = phemo(temb)
    print(f"  (c) SHIPPED top-100                 n={len(tseq)} "
          f"ESM2 P(hemo) median={np.median(tph):.3f} mean={tph.mean():.3f} "
          f"frac>0.5={(tph>0.5).mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
