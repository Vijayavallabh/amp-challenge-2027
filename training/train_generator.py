"""Train one peptide generator on one GPU. Fan out across the 8 H100s for an ensemble.

Run (single model):
    uv run --group train python training/train_generator.py --gpu 0 --seed 0 --out training/runs/gen0

The launcher (launch_ensemble.py) starts one of these per GPU with different seeds.

Design notes
------------
* All of the corpus is trainable, but we hold out a small validation split (seeded) purely
  to watch for over-fitting -- a memorizing model is useless here because the corpus is
  also the novelty screen reference.
* Every eval we sample and measure how many samples are *novel* (not verbatim in the
  training set). If novelty collapses, the model is memorizing and the config needs to
  change (more dropout, smaller model, fewer epochs). This is printed and logged.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amp_challenge_2027.data import training_sequences  # noqa: E402
from amp_challenge_2027.nn import (  # noqa: E402
    PAD, LMConfig, MAX_LEN, PeptideLM, encode, sample,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_tensors(sequences: list[str]) -> torch.Tensor:
    """[N, MAX_LEN] padded token tensor."""
    rows = torch.full((len(sequences), MAX_LEN), PAD, dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids = encode(seq)
        rows[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return rows


def lm_loss(model: PeptideLM, batch: torch.Tensor) -> torch.Tensor:
    logits = model(batch[:, :-1])
    target = batch[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=PAD
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--sample-n", type=int, default=2000)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    sequences = training_sequences()
    train_set = set(sequences)
    data = build_tensors(sequences)

    # Seeded train/val split.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(data), generator=g)
    n_val = int(len(data) * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_data = data[train_idx].to(device)
    val_data = data[val_idx].to(device)

    cfg = LMConfig(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                   d_ff=args.d_model * 4, dropout=args.dropout)
    model = PeptideLM(cfg).to(device)
    n_params = model.num_params()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    steps_per_epoch = max(1, len(train_data) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_frac)

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    log_path = args.out / "train.log"
    metrics_path = args.out / "metrics.json"
    best_val = float("inf")
    history = []
    step = 0
    t0 = time.time()

    def log(msg: str) -> None:
        line = f"[gen seed={args.seed} gpu={args.gpu}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    log(f"params={n_params/1e6:.2f}M cfg={cfg.as_dict()} "
        f"train={len(train_data)} val={len(val_data)} steps/epoch={steps_per_epoch}")

    for epoch in range(args.epochs):
        model.train()
        eg = torch.Generator(device="cpu").manual_seed(args.seed * 100000 + epoch)
        order = torch.randperm(len(train_data), generator=eg)
        epoch_loss = 0.0
        for b in range(steps_per_epoch):
            idx = order[b * args.batch_size : (b + 1) * args.batch_size]
            batch = train_data[idx]
            for pg in opt.param_groups:
                pg["lr"] = args.lr * lr_at(step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = lm_loss(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            step += 1
        epoch_loss /= steps_per_epoch

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                vloss = 0.0
                vb = 0
                for b in range(max(1, len(val_data) // args.batch_size)):
                    batch = val_data[b * args.batch_size : (b + 1) * args.batch_size]
                    if len(batch) == 0:
                        continue
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        vloss += lm_loss(model, batch).item()
                    vb += 1
                vloss /= max(1, vb)
            # Novelty probe: sample and measure how many are NOT verbatim training peptides.
            sg = torch.Generator(device=device).manual_seed(args.seed)
            samples = sample(model, args.sample_n, device=device, temperature=1.0,
                             generator=sg, batch_size=args.sample_n)
            uniq = set(samples)
            novel = sum(1 for s in uniq if s not in train_set)
            valid = sum(1 for s in samples if 8 <= len(s) <= 50)
            novelty = novel / max(1, len(uniq))
            uniq_frac = len(uniq) / max(1, len(samples))
            mins = min((len(s) for s in samples), default=0)
            maxs = max((len(s) for s in samples), default=0)
            elapsed = time.time() - t0
            log(f"epoch {epoch+1}/{args.epochs} train_loss={epoch_loss:.4f} val_loss={vloss:.4f} "
                f"novelty={novelty:.3f} uniq={uniq_frac:.3f} valid={valid/len(samples):.3f} "
                f"len[{mins},{maxs}] {elapsed:.0f}s")
            history.append(dict(epoch=epoch + 1, train_loss=epoch_loss, val_loss=vloss,
                                novelty=novelty, uniq_frac=uniq_frac,
                                valid_frac=valid / len(samples), elapsed=elapsed))
            if vloss < best_val:
                best_val = vloss
                torch.save({"state_dict": model.state_dict(), "config": cfg.as_dict(),
                            "seed": args.seed, "val_loss": vloss, "epoch": epoch + 1},
                           args.out / "best.pt")
            with open(metrics_path, "w") as fh:
                json.dump(dict(seed=args.seed, n_params=n_params, config=cfg.as_dict(),
                               best_val=best_val, history=history), fh, indent=2)

    # Always save the final model too.
    torch.save({"state_dict": model.state_dict(), "config": cfg.as_dict(),
                "seed": args.seed, "val_loss": best_val, "epoch": args.epochs},
               args.out / "final.pt")
    log(f"done. best_val={best_val:.4f} total={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
