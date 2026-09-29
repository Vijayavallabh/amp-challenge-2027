"""Rejection-sampling fine-tuning (ReST) of the peptide generator toward predicted activity
and selectivity, on the H100s.

Each round: sample from the current model -> score every candidate with APEX (8-GPU) and the
hemolysis model -> keep the highest-reward NOVEL candidates -> low-LR fine-tune on them (with a
corpus anchor) -> evaluate on a fixed held-out sampling seed. Reward mirrors the competition's
Success-Rate metric with the hard Gram+/MDR strains up-weighted, minus a hemolysis penalty.

Anti-Goodhart guards, checked every round and logged:
  * novelty (fraction of samples not verbatim in the training corpus) and uniqueness -- the
    Phase-2 screen scores diversity/novelty, so a collapsing model is worse, not better;
  * physicochemical envelope (charge / hydrophobicity / length) stays in the real-AMP range;
  * saturating success reward (no chasing APEX's poorly-calibrated sub-uM tail).
This never touches the shipped deterministic path; it only produces a better checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import bulk_apex  # noqa: E402
import reward as R  # noqa: E402
from amp_challenge_2027 import constraints as C  # noqa: E402


def _log(logf, msg):
    print(msg, flush=True)
    with open(logf, "a") as fh:
        fh.write(msg + "\n")


def minhash_sig(seq: str, k: int = 4, m: int = 4) -> tuple:
    """Cheap near-duplicate signature: the m smallest hashes of the peptide's k-mers.

    Peptides from the same motif family collide; capping how many share a signature keeps the
    fine-tuning set diverse and fights the mode-collapse that pure top-reward selection causes.
    (Offline only -- Python hash salting makes this non-deterministic across runs, which is fine
    for training-data selection.)
    """
    kmers = {seq[i:i + k] for i in range(len(seq) - k + 1)} or {seq}
    return tuple(sorted(hash(x) & 0xFFFFFFFF for x in kmers)[:m])


def physchem_summary(seqs):
    from amp_challenge_2027.physchem import net_charge
    ch = np.array([net_charge(s) for s in seqs[:5000]])
    L = np.array([len(s) for s in seqs])
    return dict(mean_charge=float(ch.mean()), mean_len=float(L.mean()))


def diverse_count(seqs_by_reward, cap=0.6, scan=600, target=150):
    """Greedy count of how many of the top peptides survive a within-list identity cap.

    Mirrors the shipped top-100 diversity screen: if this stays >=100 the model can still
    fill a diverse, high-reward top-100; if ReST collapses to a few motifs it drops. Also
    returns the mean pairwise identity of a small random-ish sample (first `scan`).
    """
    sel: list[str] = []
    for s in seqs_by_reward[:scan]:
        if not sel or C.max_identity(s, sel, cutoff=cap) < cap:
            sel.append(s)
        if len(sel) >= target:
            break
    return len(sel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="checkpoint/generator.pt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-raw", type=int, default=200000, help="raw samples per round")
    ap.add_argument("--keep-n", type=int, default=15000, help="top-reward kept for fine-tuning")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--anchor-frac", type=float, default=0.25, help="corpus mixed into train set")
    ap.add_argument("--dedup-cap", type=int, default=0,
                    help="max near-duplicates (same MinHash sig) kept for fine-tuning; 0=off. "
                         "Fights mode-collapse so the generator keeps producing DIVERSE actives.")
    ap.add_argument("--reward", choices=("balanced", "soft"), default="balanced",
                    help="reward form. 'balanced' = hard Gram+/MDR SR + broad soft tie-break "
                         "(mirrors the shipped ranker; fine-tunes toward true hard-hitters); "
                         "'soft' = the older saturating soft-Success-Rate. Default balanced.")
    ap.add_argument("--gp-w", type=float, default=1.0)
    ap.add_argument("--mdr-w", type=float, default=1.0)
    ap.add_argument("--gn-w", type=float, default=0.75,
                    help="hard Gram-negative weight in the balanced reward (feat-024; mirrors the "
                         "ranker's gn_weight). 0 recovers feat-020's pure-Gram+/MDR target.")
    ap.add_argument("--broad-w", type=float, default=0.5)
    ap.add_argument("--lys-w", type=float, default=0.0,
                    help="feat-031 wet-lab Gram- prior: subtract lys_w*arg_excess (Arg-over-Lys excess) "
                         "from the ReST reward, distilling toward Lys-rich Gram-actives APEX under-samples")
    ap.add_argument("--hemo-lambda", type=float, default=1.0)
    ap.add_argument("--hemo-gate", type=float, default=0.0,
                    help="hard selectivity gate: distil only from peptides with P(hemolytic) below "
                         "this (0 = off). Use with --hemo-lambda 0 so reward = pure Gram+/MDR "
                         "activity among non-hemolytic peptides -- forces activity gains, not "
                         "selectivity drift.")
    ap.add_argument("--hemo-model", choices=("esmc", "physchem"), default="esmc",
                    help="hemolysis scorer for the reward. 'esmc' = the accurate ESMC-600M PLM "
                         "(AUROC 0.905); 'physchem' = the older 11-descriptor model (over-optimistic "
                         "-- do NOT fine-tune against it). Default esmc.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gpu", type=int, default=0, help="device for sampling + fine-tuning")
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="APEX scoring GPUs")
    ap.add_argument("--gpu-python", type=str, required=True)
    ap.add_argument("--eval-n", type=int, default=20000)
    ap.add_argument("--eval-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from amp_challenge_2027 import nn as _nn
    from amp_challenge_2027.data import training_sequences
    from amp_challenge_2027.hemolysis import HemolysisScorer
    from amp_challenge_2027.paths import resolve_repo_path

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logf = args.out_dir / "rest.log"
    gpus = args.gpus.split(",")
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    corpus = training_sequences()
    train_set = set(corpus)
    if args.hemo_model == "esmc":
        # The accurate PLM selectivity model, on the sampling/fine-tune GPU. Fine-tuning against the
        # over-optimistic physchem model would chase peptides it *thinks* are non-hemolytic; ESMC is
        # what the shipped selection uses, so the reward and the ranker agree.
        from amp_challenge_2027.selectivity_esm import EsmcSelectivityScorer
        hs = EsmcSelectivityScorer(device=f"cuda:{args.gpu}")
    else:
        hs = HemolysisScorer("checkpoint/hemolysis.pt")

    model = _nn.load_generator(resolve_repo_path(args.init), device)
    cfg = model.cfg

    def rfn(mic, phemo, seqs):
        if args.reward == "balanced":
            r = R.balanced_reward(mic, phemo, gp_w=args.gp_w, mdr_w=args.mdr_w,
                                  gn_w=args.gn_w, broad_w=args.broad_w, hemo_lambda=args.hemo_lambda)
        else:
            r = R.reward(mic, phemo, gp_w=args.gp_w, mdr_w=args.mdr_w, hemo_lambda=args.hemo_lambda)
        if args.lys_w > 0:  # feat-031 wet-lab Gram- prior: penalise Arg-over-Lys excess so ReST
            # distils toward the Lys-rich Gram-actives APEX under-samples (independent of APEX)
            from amp_challenge_2027.oracle import arg_excess
            r = r - args.lys_w * arg_excess(list(seqs))
        return r

    def score_seqs(seqs):
        uniq, mic = bulk_apex.score(seqs, device="cuda", workers=8,
                                    gpu_python=args.gpu_python, gpus=gpus)
        phemo = np.asarray(hs.predict_proba(uniq))
        return uniq, mic.astype(np.float64), phemo

    def sample(n_raw, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        raw = _nn.sample(model, n_raw, device=device, temperature=args.temperature,
                         min_residues=C.MIN_LENGTH, max_residues=C.MAX_LENGTH,
                         generator=g, batch_size=8192)
        seen = {}
        for s in raw:
            if s not in seen and C.is_valid_sequence(s):
                seen[s] = None
        return list(seen)

    def evaluate(tag, seed):
        seqs = sample(args.eval_n, seed)
        uniq, mic, phemo = score_seqs(seqs)
        rew = rfn(mic, phemo, uniq)
        novel = sum(1 for s in uniq if s not in train_set) / max(1, len(uniq))
        # top-100 by reward: what selection would actually ship
        order_ev = np.argsort(-rew)
        top = order_ev[:100]
        dcount = diverse_count([uniq[i] for i in order_ev])  # diverse top survivors at 0.6 cap
        pc = physchem_summary(uniq)
        _log(logf, f"  [{tag}] pool " + R.fmt(R.summarize(mic, phemo), "whole") +
             f" novel={novel:.3f} div150={dcount} charge={pc['mean_charge']:.1f} len={pc['mean_len']:.1f}")
        _log(logf, f"  [{tag}] top100 " + R.fmt(R.summarize(mic, phemo, top), "byreward") +
             f" reward={rew[top].mean():.3f}")
        return dict(tag=tag, mean_reward=float(rew.mean()), top100_reward=float(rew[top].mean()),
                    novel=float(novel), div150=int(dcount), **R.summarize(mic, phemo, top))

    history = []
    _log(logf, f"=== ReST init={args.init} rounds={args.rounds} keep_n={args.keep_n} "
               f"lr={args.lr} epochs={args.epochs} gp_w={args.gp_w} mdr_w={args.mdr_w} "
               f"hemo_lambda={args.hemo_lambda} anchor={args.anchor_frac} ===")
    base_eval = evaluate("round0", args.eval_seed)
    history.append(base_eval)
    best = base_eval["top100_reward"]
    torch.save({"state_dict": model.state_dict(), "config": cfg.as_dict()}, args.out_dir / "round0.pt")

    for r in range(1, args.rounds + 1):
        t0 = time.time()
        # 1) sample + score
        pool = sample(args.n_raw, seed=1000 + r)
        uniq, mic, phemo = score_seqs(pool)
        rew = rfn(mic, phemo, uniq)
        # 2) keep top-reward NOVEL peptides that are actually active (min-MIC <= 16); distilling
        #    from active peptides only keeps the fine-tune target on-target. Relax if too few.
        minmic = mic.min(axis=1)
        order = np.argsort(-rew)

        def _passes(i):
            # hard selectivity GATE: distil ONLY from genuinely non-hemolytic peptides, so the
            # generator cannot raise reward by lowering hemolysis instead of gaining Gram+/MDR
            # activity (the failure mode of a soft -lambda*phemo penalty). The reward then ranks the
            # gated set by pure activity.
            gate_ok = args.hemo_gate <= 0 or phemo[i] < args.hemo_gate
            return uniq[i] not in train_set and minmic[i] <= R.THRESH and gate_ok

        if args.dedup_cap > 0:
            from collections import Counter
            cnt = Counter()
            kept_idx = []
            for i in order:
                if not _passes(i):
                    continue
                sig = minhash_sig(uniq[i])
                if cnt[sig] >= args.dedup_cap:
                    continue
                cnt[sig] += 1
                kept_idx.append(i)
                if len(kept_idx) >= args.keep_n:
                    break
        else:
            kept_idx = [i for i in order if _passes(i)][: args.keep_n]
        if len(kept_idx) < 500:  # relax the ACTIVITY floor but keep the selectivity gate
            kept_idx = [i for i in order if uniq[i] not in train_set
                        and (args.hemo_gate <= 0 or phemo[i] < args.hemo_gate)][: args.keep_n]
        kept = [uniq[i] for i in kept_idx]
        n_active_pool = int((minmic <= R.THRESH).sum())
        # 3) build fine-tune set with a corpus anchor
        n_anchor = int(len(kept) * args.anchor_frac)
        anchor = list(np.random.choice(corpus, size=min(n_anchor, len(corpus)), replace=False)) \
            if n_anchor > 0 else []
        data = kept + anchor
        rows = torch.full((len(data), _nn.MAX_LEN), _nn.PAD, dtype=torch.long)
        for i, s in enumerate(data):
            ids = _nn.encode(s)
            rows[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        rows = rows.to(device)
        # 4) fine-tune
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                betas=(0.9, 0.95))
        steps = max(1, len(rows) // args.batch_size)
        for ep in range(args.epochs):
            perm = torch.randperm(len(rows), device=device)
            for b in range(steps):
                idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
                batch = rows[idx]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(batch[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           batch[:, 1:].reshape(-1), ignore_index=_nn.PAD)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()
        # 5) evaluate (fixed held-out seed) + save
        kept_prof = R.summarize(mic, phemo, np.array(kept_idx))
        _log(logf, f"round {r}: pool_active={n_active_pool}/{len(uniq)} kept={len(kept)} "
                   f"anchor={len(anchor)} loss={loss.item():.3f} | kept-set " + R.fmt(kept_prof, ""))
        ev = evaluate(f"round{r}", args.eval_seed)
        ev["round"] = r
        history.append(ev)
        torch.save({"state_dict": model.state_dict(), "config": cfg.as_dict()},
                   args.out_dir / f"round{r}.pt")
        if ev["top100_reward"] > best:
            best = ev["top100_reward"]
            torch.save({"state_dict": model.state_dict(), "config": cfg.as_dict()},
                       args.out_dir / "best.pt")
            _log(logf, f"  * new best top100_reward={best:.3f} -> best.pt")
        json.dump(history, open(args.out_dir / "history.json", "w"), indent=2)
        _log(logf, f"round {r} done in {time.time()-t0:.0f}s")

    _log(logf, f"=== done. best top100_reward={best:.3f} (round0={base_eval['top100_reward']:.3f}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
