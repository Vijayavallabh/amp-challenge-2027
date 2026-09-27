# Model plan — the competitive pipeline

The goal is peptides that score well where the competition actually measures: wet-lab **MIC**
(potency across 20 strains) and **HC50** (low hemolysis → selectivity), on 25 peptides drawn at
random from each team's **top 50**, while the 50k library passes phase-1 screening on diversity,
novelty and physicochemical distributions. No approach can *guarantee* winning — biology and other
teams decide that — so this plan maximizes the levers we control and uses the 8× H100s hard.

## What the score rewards, and the levers

1. **Top-100 ranking quality is paramount.** 25 of the top 50 are synthesized and assayed, and
   scoring is the mean over tested peptides. A strong **activity + hemolysis oracle** that ranks
   genuinely potent, non-hemolytic, novel peptides to the top is at least as important as the
   generator. This is where most of the win is.
2. **Novelty is a hard gate.** Top-100 must stay ≤80% Levenshtein identity to the reference set;
   the library may contain no exact match. The generator must generalize, not memorize.
3. **Library diversity/novelty/physchem** decides phase-1 advancement (top 20 proceed).

## Architecture

```
corpus (feat-005)  ──►  Generator (feat-006)  ──►  huge candidate pool
   │                         ▲                            │
   │                         │ property conditioning      ▼
   └──► labeled data ──► Oracle ensemble (feat-007) ──► score: activity + selectivity
        (MIC, HC50)                                        │
                                                           ▼
                              novelty screen + diversity selection
                                                           │
                                            ┌──────────────┴──────────────┐
                                            ▼                             ▼
                                  library.fasta (50k)            top.fasta (100, ranked)
```

### A. Generator (feat-006)

- **Shipped model:** an **autoregressive Transformer LM over the 20-AA alphabet**, defined inside
  the package (`nn.py`) so inference needs only `torch` — no `transformers` at generate time — and
  is fully deterministic and CPU-runnable. Trained from scratch on the corpus; novelty comes from
  temperature / nucleus sampling. **Property-conditioned** (length, net charge bucket, and later a
  predicted-activity bucket) so generation can be steered toward active-like peptides.
- **Exploration models (H100-heavy, offline only):** fine-tune a pretrained protein LM (ESM-2
  650M/3B) as a second generator and as the oracle backbone. Used to enrich/relabel the candidate
  pool offline; their *outputs* (selected sequences) feed selection, so they are not runtime deps.
- **Scale & compute use:** train large, long, and as an ensemble across seeds/sizes; run continuous
  directed-evolution / guided sampling against the oracle until the deadline.

### B. Oracle ensemble (feat-007)

- Fine-tune **ESM-2** into (a) an antibacterial-activity / MIC head and (b) a hemolysis / HC50 head,
  as an ensemble across seeds and sizes for robust ranking. This is the ranking function for the
  top-100 and the guide for directed generation.
- **Needs labeled data** (feat-007a) — MIC per strain (DBAASP) and hemolysis (HemoPI, DBAASP
  hemolytic). Licensing diligence required: everything used must be publicly redistributable to keep
  co-authorship eligibility. Each source gets a row in `docs/DATA.md` with its terms before use.

### C. Selection / optimization

- Oversample millions of candidates → filter to valid + novel (≤80% top, no exact match library) →
  score with the oracle ensemble → select a **diverse** high-activity/low-hemolysis top 100 and a
  diverse, novel 50k library. Diversity enforced by identity-clustering so the top 100 isn't 100
  near-duplicates (the random-draw scoring punishes redundancy).

### D. Reproducible `generate` (feat-006b)

- Train offline on GPU; save weights to `checkpoint/` (safetensors). `uv run generate` loads them,
  samples **deterministically on CPU** (fixed seed, `torch.use_deterministic_algorithms(True)`),
  ranks with the shipped scorer, writes both files. Runtime deps: `torch` (+ `safetensors`).
- The validator's reproducibility check runs `generate` twice on one machine and compares bytes;
  a fixed seed + deterministic CPU inference satisfies it. Generation of 50k short sequences on CPU
  is minutes — acceptable and portable to a CPU-only validator box.
- The submission stays valid throughout: `build_model()` keeps returning `RandomBaseline` until the
  trained generator passes the full gate (`./init.sh`), the official validator, and a novelty/
  diversity check on real output. Only then is the swap committed.

## Compute environment

8× H100 80GB, local (this host is the DGX; no SSH — see the `env-gpu-is-local-8xh100` note).
torch 2.10+cu128, transformers 5.16, accelerate, datasets present. Train with DDP/accelerate across
all 8 GPUs; launch in local tmux or background; monitor with `nvidia-smi`.

## Feature breakdown

- **feat-006a** generator training scaffold + a from-scratch AR Transformer, trained on all 8 GPUs.
- **feat-006b** deterministic CPU sampling wired into `generate`; swap `build_model()`; stays valid.
- **feat-006c** scale up + property conditioning + ensemble.
- **feat-007a** acquire & disclose labeled MIC/HC50 data (licensing diligence).
- **feat-007b** oracle ensemble (ESM-2 fine-tunes) for activity + hemolysis.
- **feat-007c** oracle-guided selection & diversity for the top-100 and library.

## Honesty guardrails

- Never claim a peptide "is" active — only "predicted active" by the oracle, with its measured
  validation metrics stated.
- Report oracle accuracy/AUROC and the novelty-rejection rate; if the model mostly reproduces known
  AMPs, say so and fix it rather than hiding it behind the screen.
- Keep the shipped runtime lean and deterministic; heavy models stay in the offline/train path.
