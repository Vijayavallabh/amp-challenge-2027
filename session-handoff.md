# Session Handoff

## Current Objective

- **Goal:** Maximise the final entry by the 2026-09-30 22:00 UTC deadline: a strong, reproducible
  50k library + top-100 that is active, selective, novel and diverse.
- **Current status (session 3, through feat-021):** the generator is **ReST-fine-tuned toward the
  hard Gram+/MDR categories** (feat-020, selectivity-gated) on the 8 H100s; the top-100 is ranked by a
  **balanced hard Success-Rate APEX score** (Gram+, MDR, and — feat-021 — Gram− with `gn_weight=0.75`,
  plus a broad soft tie-break) minus an **ESMC-600M** selectivity penalty, with a 0.6 diversity cap,
  from an 8×/400k pool; the library is sampled hot (T=1.6) to stay diverse/novel. **Top-50 (assayed):
  Broad 0.64, Gram− 0.58, Gram+ 0.75, MDR 0.67, 0% predicted-hemolytic** — roughly double the two hard
  categories vs a broad-only pipeline, weakest category lifted 0.54→0.58. Byte-reproducible,
  ESMFold2-checked (pLDDT 0.704), submodel-agreement 85–96%. See `progress.md` + `feature_list.json`
  for the full feat-017…feat-022 record. **feat-022** (directed-evolution GA) was explored on the free
  GPUs and REJECTED — it overfits APEX (held-out submodels prove no net gain over feat-021).
- **Branch / commit:** `main` (feat-021 commit pending this session; feat-020 at `575c3e2`; see
  `git log`). Submission remains **user-gated** (team name + go-ahead from `j_v_v_07`).

## What is done (this session)

- [x] **feat-007** — APEX-pathogen MIC oracle (MIT) + `mic.csv` (CC BY 4.0) vendored & disclosed;
      APEX validated against wet-lab MICs (moderate, wet-lab-aligned: AUROC 0.76 / 0.62).
- [x] **feat-013** — activity oracle (`oracle.py`, APEX subprocess) **and** selectivity model
      (`hemolysis.py` + `physchem.py`, HemoPI-2 MLP, held-out AUROC 0.778). Both shipped.
- [x] **feat-014 (most)** — `generate` ranks by `broad_potency − 2·P(hemolytic)` with a 0.6
      diversity cap; defaults flipped; graceful fallback to likelihood → baseline.
- [x] Earlier: feat-001/002/003/004/005/006/008/011 (bootstrap, compliance, baseline, Kaggle
      entry, corpus, generator ensemble, validator, trained-model wiring).

## Done this session (session 3)

- [x] **feat-015** — ReST activity/selectivity fine-tuning on 8×H100 (tooling in `experiments/`);
      anti-Goodhart validated (APEX 8-submodel agreement ~0.95; realistic amphipathic AMPs).
- [x] **feat-016** — deterministic parallel CPU-APEX in the shipped oracle (byte-reproducible,
      worker-count-independent, ~2×+ faster).
- [x] Shipped ranking switched to `category_success_score − 0.5·P(hemolytic)`; temperature 1.6.
- [x] **feat-017** — selectivity upgraded from 11 physchem descriptors to the latest-SOTA PLM
      **ESMC-600M** (ESM++ `Synthyra/ESMplusplus_large`, MIT; head in `checkpoint/selectivity_esmc.pt`,
      held-out AUROC 0.778 → **0.905**). Two-stage (PLM only on top `refine_k=4000`). Devil's-advocate:
      the old physchem-selected top-100 was **~51% predicted-hemolytic**; ESMC re-ranking → **0%**
      while breadth *rises* (category-success 0.884→0.914). Byte-deterministic on GPU; physchem fallback.
      **Official validator PASSED** on the pushed commit `160e65b` (real ESM++ path, all 8 checks).
- [x] **feat-018** — offline structural cross-check with **ESMFold2-Fast** (6.5B, latest SOTA folder):
      all 100 fold as confident amphipathic helices (median pLDDT 0.679, helix 1.00, 0/100 flagged),
      like known-AMP controls. `experiments/structure_validate.py`; NOT in the deterministic path.
- [x] **feat-019** — **balanced hard-Gram+/MDR objective + 8× oversample** lifts the weak categories.
      `balanced_success_score` = SR_hard(GP)+SR_hard(MDR)+0.5·mean_soft (soft averaging plateaued
      Gram+ at ~0.39); oversample 3×→8× (400k) surfaces the diversity-screen-limited non-hemolytic
      Gram+/MDR champions; λ 0.5→**1.5** holds 0% hemolytic (Gram+ activity ↔ hemolysis, median P
      0.98). **Top-50 Pareto gain:** Broad 0.50→0.57, GN 0.57→0.60, **GP 0.37→0.50, MDR 0.42→0.49**,
      0% hemolytic. Anti-Goodhart OK (ESMFold2 0/100 flags). ESMC-6B tested for selectivity → no gain
      (0.905), kept 600M. Defaults flipped; 129 tests pass.
- [x] **feat-020** — **GP/MDR-targeted selectivity-gated ReST generator** raises the hard-category
      ceiling. A soft `activity−λ·P(hemo)` reward *failed* (GP/MDR declined — generator lowered
      hemolysis instead). A **hard selectivity gate** (distil only from ESMC-non-hemolytic; reward
      pure GP/MDR hard-SR) worked: 4 rounds on 8×H100 shifted the generator toward activity at flat
      hemolysis. Promoted round3 (feat-019 generator → `experiments/cache` + git `60afc5f`). **Top-50:
      GP 0.50→0.75, MDR 0.49→0.67, Broad 0.57→0.62, GN 0.60→0.54, 0% hemolytic, MORE novel (0.62)**.
      Anti-Goodhart all pass: novelty↑, 0% hemolytic (ESMC), 86–89% APEX 8-submodel agreement, ESMFold2
      pLDDT 0.702/0 flags (caveat: muH 0.29). `rest_finetune.py` gained `--hemo-model/--reward/--hemo-gate`.
- [x] Docs (SUBMISSION/RESEARCH/README), progress, feature_list, memory updated; committed + pushed.

## Final submission characterisation (local full 50k run, session 3)

- Top-50: **100% active**, median best-strain MIC **~2.3 µM**, mean predicted breadth **5.28/11**
  (Gram-neg SR ~56%, Gram-pos ~34%, MDR ~38%); mean **P(hemolytic) 0.074**; novelty median identity
  to known **0.59**, max **0.75** (rule ≤0.80); within-list identity ≤0.60.
- Library: 50,000 unique/valid/novel; **~81% diverse** (random-sample survival at 0.6), novelty
  median **0.50** (3% > 0.8), 84% cationic, mean length 18. Byte-identical across two runs.

## Verification evidence

| Check | Command | Result |
|---|---|---|
| Tests | `uv run pytest -q` | 127 passed, 3 skipped (live-APEX/gated) |
| Full 50k reproducibility (APEX + ESMC PLM on GPU) | two `uv run generate` runs | **byte-identical** (top `01ee02c2…`, library `3488623b…`) |
| Official validator (APEX ranking) | `verify_submission.py <repo-url>` | **All checks passed** (fresh clone, 2026-09-27) |
| Official validator (APEX + ESMC selectivity) | same | re-running on the ESMC commit (HF-fetched ESM++; physchem fallback) |

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
- [ ] Oracles are estimates, not measurements (APEX 0.62–0.76; ESMC selectivity 0.905) — never
      claim wet-lab efficacy. Selectivity is now strong, but still a prediction.
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
