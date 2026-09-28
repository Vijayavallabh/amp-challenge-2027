"""Sample a candidate pool from a generator checkpoint (GPU), filter, and write seqs.

Offline tool. Draws raw samples from the AR-Transformer, keeps unique valid sequences, and
(optionally) drops any that are verbatim training peptides (the novelty screen reference).
Writes one sequence per line for scoring by bulk_apex.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from amp_challenge_2027 import constraints as C  # noqa: E402


def sample_pool(checkpoint: str, n_raw: int, *, gpu: int = 0, temperature: float = 1.0,
                top_p: float = 1.0, seed: int = 0, exclude: set[str] | None = None,
                batch_size: int = 8192) -> list[str]:
    import torch

    from amp_challenge_2027 import nn as _nn
    from amp_challenge_2027.paths import resolve_repo_path

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = _nn.load_generator(resolve_repo_path(checkpoint), device)
    g = torch.Generator(device=device).manual_seed(seed)
    raw = _nn.sample(model, n_raw, device=device, temperature=temperature, top_p=top_p,
                     min_residues=C.MIN_LENGTH, max_residues=C.MAX_LENGTH,
                     generator=g, batch_size=batch_size)
    exclude = exclude or set()
    seen: dict[str, None] = {}
    for s in raw:
        if s in seen or s in exclude:
            continue
        if C.is_valid_sequence(s):
            seen[s] = None
    return list(seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoint/generator.pt")
    ap.add_argument("--n-raw", type=int, default=200000, help="raw samples to draw")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--novel-only", action="store_true", help="drop verbatim training peptides")
    args = ap.parse_args()

    exclude = None
    if args.novel_only:
        from amp_challenge_2027.data import training_sequences
        exclude = set(training_sequences())

    pool = sample_pool(args.checkpoint, args.n_raw, gpu=args.gpu, temperature=args.temperature,
                       top_p=args.top_p, seed=args.seed, exclude=exclude)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(pool) + "\n")
    print(f"raw={args.n_raw} -> unique_valid{'_novel' if args.novel_only else ''}={len(pool)} "
          f"({100*len(pool)/args.n_raw:.1f}%) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
