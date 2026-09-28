"""Train the ESM-2-embedding selectivity (hemolysis) model shipped for top-100 re-ranking.

Runs in the shipped env (torch + fair-esm) so the embeddings match inference exactly. Embeds
HemoPI-2 with ESM-2 (esm2_t30_150M), trains a small MLP head, reports held-out AUROC, and saves
checkpoint/selectivity_esm.pt {state_dict, mu, sd, esm_model, hidden, val_auroc}.

    uv run python training/train_selectivity_esm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
HEMO = REPO / "data" / "hemolysis" / "HemoPI2.fasta"
OUT = REPO / "checkpoint" / "selectivity_esm.pt"
ESM_MODEL = "esm2_t30_150M_UR50D"
LAYER = 30


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


@torch.no_grad()
def embed(seqs, model, alphabet, device, batch=64):
    bc = alphabet.get_batch_converter()
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        _, _, toks = bc([(str(j), s) for j, s in enumerate(chunk)])
        rep = model(toks.to(device), repr_layers=[LAYER])["representations"][LAYER]
        for k, s in enumerate(chunk):
            out.append(rep[k, 1:len(s) + 1].mean(0).float().cpu().numpy())
    return np.array(out, dtype=np.float32)


def main():
    import esm
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = parse()
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.eval().to(device)
    Xtr = embed([s for s, _ in tr], model, alphabet, device)
    Xva = embed([s for s, _ in va], model, alphabet, device)
    ytr = np.array([y for _, y in tr], dtype=np.float32)
    yva = np.array([y for _, y in va], dtype=np.float32)

    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr_n = torch.tensor((Xtr - mu) / sd); Xva_n = torch.tensor((Xva - mu) / sd)
    ytr_t = torch.tensor(ytr)

    hidden = 128
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    best_auroc, best_state = 0.0, None
    for epoch in range(300):
        net.train()
        opt.zero_grad()
        loss = lossf(net(Xtr_n).squeeze(1), ytr_t)
        loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                p = torch.sigmoid(net(Xva_n).squeeze(1)).numpy()
            au = roc_auc_score(yva, p)
            if au > best_auroc:
                best_auroc = au
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd, "esm_model": ESM_MODEL,
                "layer": LAYER, "hidden": hidden, "val_auroc": float(best_auroc)}, OUT)
    print(f"ESM-2 selectivity model: held-out AUROC = {best_auroc:.3f} (shipped physchem: 0.778) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
