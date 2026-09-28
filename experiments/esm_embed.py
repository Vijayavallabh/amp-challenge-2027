"""Extract mean-pooled ESM-2 embeddings for a list of peptides (GPU), cache to npz.

Run with the ESM env python. Reused to apply an ESM-2-based selectivity/activity model to a
candidate pool (for re-ranking the top-100 or for a re-ReST reward).

    CUDA_VISIBLE_DEVICES=0 <esm-env>/bin/python experiments/esm_embed.py --in seqs.txt --out emb.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def read_seqs(path: Path) -> list[str]:
    text = path.read_text()
    if path.suffix in (".fasta", ".fa"):
        seqs, cur = [], []
        for line in text.splitlines():
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
            elif line.strip():
                cur.append(line.strip())
        if cur:
            seqs.append("".join(cur))
        return seqs
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


@torch.no_grad()
def embed(seqs, model, alphabet, layer, device, batch=128):
    bc = alphabet.get_batch_converter()
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        _, _, toks = bc([(str(j), s) for j, s in enumerate(chunk)])
        toks = toks.to(device)
        rep = model(toks, repr_layers=[layer])["representations"][layer]
        for k, s in enumerate(chunk):
            out.append(rep[k, 1:len(s) + 1].mean(0).float().cpu().numpy())
    return np.array(out, dtype=np.float32)


def main():
    import esm
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqs = read_seqs(args.inp)
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.eval().to(device)
    X = embed(seqs, model, alphabet, 30, device, args.batch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, sequences=np.array(seqs, dtype=object), emb=X)
    print(f"embedded {len(seqs)} peptides -> {X.shape} -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
