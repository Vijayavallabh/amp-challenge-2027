"""Train the shipped hemolysis/selectivity classifier on HemoPI-1.

Honest protocol: fit on the HemoPI-1 *main* split (pm/nm), report AUROC/accuracy on the
designated held-out *validation* split (pv/nv) that is never seen in training. Deterministic
(fixed seed, fixed epoch budget), so the shipped `checkpoint/hemolysis.pt` is reproducible.

    uv run --group train python training/train_hemolysis.py

Writes `checkpoint/hemolysis.pt`: state_dict + the feature standardisation fitted on main +
the held-out metrics. Inference is `amp_challenge_2027.hemolysis.HemolysisScorer`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from amp_challenge_2027.constraints import is_valid_sequence
from amp_challenge_2027.fasta import read_fasta
from amp_challenge_2027.hemolysis import build_mlp
from amp_challenge_2027.paths import resolve_repo_path
from amp_challenge_2027.physchem import FEATURE_NAMES, feature_matrix

# HemoPI-2: high vs low hemolytic potency, both from real Hemolytik peptides. Chosen over
# HemoPI-1 because HemoPI-1's negatives are random SwissProt fragments, so a model trained on it
# learns "AMP-like -> hemolytic" and cannot rank hemolysis *among* active AMPs (verified: it
# scored 99% of our APEX-active peptides as hemolytic). HemoPI-2 discriminates the degree.
DATA = "data/hemolysis/HemoPI2.fasta"
OUT = "checkpoint/hemolysis.pt"
HIDDEN = 16
EPOCHS = 300
LR = 0.02
WEIGHT_DECAY = 1e-3
SEED = 0


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    order = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((order[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def load_split() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    headers, seqs = read_fasta(resolve_repo_path(DATA))
    main_s, main_y, val_s, val_y = [], [], [], []
    for h, s in zip(headers, seqs):
        if not is_valid_sequence(s):  # standard 20 AA, length 8-50 (our emit domain)
            continue
        tok = h.split("_")[1]  # pm / pv / nm / nv
        y = 1 if tok[0] == "p" else 0
        (val_s if tok[1] == "v" else main_s).append(s)
        (val_y if tok[1] == "v" else main_y).append(y)
    return main_s, np.array(main_y, float), val_s, np.array(val_y, float)


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    main_s, main_y, val_s, val_y = load_split()
    Xtr, Xva = feature_matrix(main_s), feature_matrix(val_s)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_n = torch.tensor((Xtr - mean) / std, dtype=torch.float32)
    Xva_n = torch.tensor((Xva - mean) / std, dtype=torch.float32)
    ytr = torch.tensor(main_y, dtype=torch.float32)

    model = build_mlp(len(FEATURE_NAMES), HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = torch.nn.BCEWithLogitsLoss()
    print(f"train(main)={len(main_s)}  held-out(val)={len(val_s)}  features={len(FEATURE_NAMES)}")
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xtr_n).squeeze(1), ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        tr = torch.sigmoid(model(Xtr_n).squeeze(1)).numpy()
        va = torch.sigmoid(model(Xva_n).squeeze(1)).numpy()
    tr_auroc, va_auroc = _auroc(main_y, tr), _auroc(val_y, va)
    va_acc = float(((va >= 0.5).astype(int) == val_y).mean())
    print(f"train AUROC={tr_auroc:.3f}  held-out AUROC={va_auroc:.3f}  held-out acc={va_acc:.3f}")

    out = Path(__file__).resolve().parents[1] / OUT   # repo root / checkpoint/hemolysis.pt
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": list(FEATURE_NAMES),
            "mean": mean, "std": std, "hidden": HIDDEN,
            "val_auroc": va_auroc, "val_acc": va_acc, "seed": SEED, "epochs": EPOCHS,
        },
        out,
    )
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
