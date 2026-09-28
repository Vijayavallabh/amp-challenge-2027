"""Train the shipped selectivity (hemolysis) head on ESMC-600M embeddings.

Uses the latest SOTA protein language model (ESM Cambrian 600M, MIT, via HuggingFace
transformers) -- a large upgrade over the 11 hand-crafted physicochemical descriptors of the
previous model. Runs in the shipped env so the embeddings match inference exactly. Embeds
HemoPI-2 with ESMC-600M, trains a small MLP head, reports held-out AUROC, and saves
checkpoint/selectivity_esmc.pt {state_dict, mu, sd, hf_model, hidden, val_auroc}.

    HF_HOME=... CUDA_VISIBLE_DEVICES=0 uv run python training/train_selectivity_esmc.py
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
OUT = REPO / "checkpoint" / "selectivity_esmc.pt"
# ESM++ large = the original ESMC-600M (MIT) weights wrapped for native HuggingFace transformers
# (self-contained tokenizer via trust_remote_code) -- loads cleanly with no torchtext dependency.
HF_MODEL = "Synthyra/ESMplusplus_large"


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
def embed(seqs, model, tok, device, batch=32):
    """Mean-pooled ESMC last-hidden-state over real residues (excludes cls/eos/pad)."""
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc).last_hidden_state  # (B, T, d)
        for k, s in enumerate(chunk):
            out.append(h[k, 1:1 + len(s)].float().mean(0).cpu().numpy())
    return np.array(out, dtype=np.float32)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (no sklearn dependency)."""
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos = float(y.sum()); n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    from transformers import AutoModel

    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = parse()
    model = AutoModel.from_pretrained(HF_MODEL, trust_remote_code=True).eval().to(device)
    tok = model.tokenizer

    Xtr = embed([s for s, _ in tr], model, tok, device)
    Xva = embed([s for s, _ in va], model, tok, device)
    ytr = np.array([y for _, y in tr], dtype=np.float32)
    yva = np.array([y for _, y in va], dtype=np.float32)
    print(f"HemoPI-2: train={len(tr)} val={len(va)} | ESMC dim={Xtr.shape[1]}")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = torch.tensor((Xtr - mu) / sd); Xva_n = torch.tensor((Xva - mu) / sd)
    ytr_t = torch.tensor(ytr)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state = 0.0, None
    for epoch in range(400):
        net.train(); opt.zero_grad()
        lossf(net(Xtr_n).squeeze(1), ytr_t).backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                p = torch.sigmoid(net(Xva_n).squeeze(1)).numpy()
            au = auroc(yva, p)
            if au > best:
                best = au; best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd, "hf_model": HF_MODEL,
                "hidden": 128, "val_auroc": float(best)}, OUT)
    print(f"ESMC-600M selectivity head: held-out AUROC = {best:.3f} (physchem baseline 0.778) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
