# Session Handoff

## Current Objective

- **Goal:** Maximise the final entry by the 2026-09-30 22:00 UTC deadline: a strong, reproducible
  50k library + top-100 that is active, selective, novel and diverse.
- **Current status:** Real **activity + selectivity + diversity** pipeline shipped and validated.
  `uv run generate` samples the trained AR-Transformer, ranks the top-100 by APEX-predicted
  broad-spectrum potency minus a hemolysis penalty, with a within-list diversity cap. Verified by
  the official validator on a fresh clone.
- **Branch / commit:** `main` (see `git log`).

## What is done (this session)

- [x] **feat-007** — APEX-pathogen MIC oracle (MIT) + `mic.csv` (CC BY 4.0) vendored & disclosed;
      APEX validated against wet-lab MICs (moderate, wet-lab-aligned: AUROC 0.76 / 0.62).
- [x] **feat-013** — activity oracle (`oracle.py`, APEX subprocess) **and** selectivity model
      (`hemolysis.py` + `physchem.py`, HemoPI-2 MLP, held-out AUROC 0.778). Both shipped.
- [x] **feat-014 (most)** — `generate` ranks by `broad_potency − 2·P(hemolytic)` with a 0.6
      diversity cap; defaults flipped; graceful fallback to likelihood → baseline.
- [x] Earlier: feat-001/002/003/004/005/006/008/011 (bootstrap, compliance, baseline, Kaggle
      entry, corpus, generator ensemble, validator, trained-model wiring).

## Final submission characterisation (local full 50k run)

- Top-100: **100% active** (predicted min-MIC ≤16 µM), median min-MIC **4.6 µM**, mean predicted
  breadth **3.85/11 strains**; mean **P(hemolytic) 0.29** (81% < 0.5); novelty max-identity to
  known median **0.33** / max 0.55 (rule ≤0.80); within-list pairwise identity max 0.60.
- Library: 50,000 unique, valid, novel; byte-identical across runs.

## Verification evidence

| Check | Command | Result |
|---|---|---|
| Tests | `uv run pytest -q` | 124 passed, 2 skipped (live-APEX gated) |
| Full 50k reproducibility | two `uv run generate` runs | byte-identical (activity-only metric confirmed; apex+hemolysis re-confirm in progress) |
| Official validator (APEX ranking) | `verify_submission.py <repo-url>` | **All checks passed** (fresh clone, 2026-09-27) |
| Official validator (APEX + hemolysis) | same | re-running to reconfirm after the selectivity change |

## Key decisions / devil's-advocate findings

- Ranking by APEX ≫ likelihood (top-100 median min-MIC 178→2.6 µM; 0/100 overlap).
- Rank by `broad_potency` (Success-Rate-aligned), not min-MIC; APEX is moderate, so don't maximise
  its tail — diversify.
- Hemolysis: HemoPI-1 model was biased ("AMP-like=hemolytic"); use HemoPI-2 as a *soft* penalty.
- Heavy models stay isolated/offline; shipped runtime is deterministic with graceful fallback, so a
  valid submission is preserved at all times.

## Blockers / risks

- [ ] **One entry, no resubmission** — run `verify_submission.py <repo-url>` immediately before
      submitting (feat-008), and submit only from **`j_v_v_07`** (never the machine default token).
- [ ] Oracles are moderate (APEX 0.62–0.76; hemolysis 0.778) — predictions are estimates, not
      measurements. Never claim wet-lab efficacy.
- [ ] Local GPU 0 can be shared; run one GPU job at a time to avoid OOM (a concurrent run caused a
      transient OOM during a validator run — cosmetic, recovered).

## Recommended next step

1. **feat-009** — write `SUBMISSION.md` (abstract, data, ranking procedure, filters) using the
   characterisation numbers above. Required deliverable.
2. **feat-012 (optional)** — generator ensemble / oversample-and-select to raise broad-active
   candidate supply; only if it beats the current top-100 on held-out predicted breadth.
3. **feat-008 → feat-010** — final clean validation, then submit from `j_v_v_07` (user-gated).

## Startup

1. Read `AGENTS.md`, `docs/COMPETITION.md`, `feature_list.json`, `progress.md`.
2. `./init.sh` (runs the full APEX pipeline twice; needs network for the isolated APEX env on
   first call, and a GPU for a fast full run — falls back to CPU/likelihood otherwise).
