# Session Progress Log

## Current State

**Last Updated:** 2026-09-28 (session 3 — ReST + parallel APEX + ESMC selectivity + balanced Gram+/MDR objective)
**Active Feature:** feat-010 SUBMIT (user-gated) — feat-015 (ReST) + feat-016 (parallel APEX) +
feat-017 (ESMC-600M selectivity) + feat-018 (ESMFold2 structure check) + feat-019 (balanced
hard-Gram+/MDR objective + 8× oversample) DONE; submission regenerated; pending final
official-validator re-run on the new commit + the participant's team name and go-ahead.
**Deadline:** 2026-09-30 22:00 UTC (1 October 2026, AOE) — see `docs/COMPETITION.md`

`uv run generate` now produces a scientifically meaningful, reproducible submission: the trained
AR-Transformer samples a novel, cationic/amphipathic 50k library, and the top-100 is ranked by
a **hard Gram+/MDR Success-Rate APEX score** (from an 8×/400k oversampled pool) **minus a λ=1.5
ESMC-600M selectivity penalty**, with a within-list diversity cap. Byte-reproducible and structurally
cross-checked (ESMFold2). Final **top-50** (the assayed set) — a clean Pareto gain over the prior
ESMC submission on all five categories: hard Success-Rate (MIC ≤16 µM) **Broad 0.57, Gram− 0.60,
Gram+ 0.50, MDR 0.49**, and **0% ESMC-predicted-hemolytic** (median P 0.004); novel (median identity
0.72, max 0.80, 0 exact), within-list identity ≤0.60. The scientific pipeline (activity + selectivity
+ diversity + structure) is in place; remaining is the final validate-and-submit (user-gated). A
GP/MDR-targeted ReST round is running on the 8 H100s as a further-optimization bet (fallback = this).

## Status

### What's Done

- [x] feat-001 — uv project bootstrap. Python pinned to 3.11, `uv.lock` committed, MIT licensed,
      `generate` wired as the console entry point.
- [x] feat-002 — Compliance layer in `src/amp_challenge_2027/constraints.py`, mirroring the
      official `scripts/verify_submission.py`. 69 tests, covering both the sequence rules and the
      repository-level rules (licence, entry point, pinned environment, defaulted arguments).
- [x] feat-004 — Kaggle entry **verified** via the API as `vijayavallabhj` / `j_v_v_07`:
      `userHasEntered=True`, 0 submissions so far.
- [x] Confirmed the competition ships **no dataset** — its only data file is a 55-byte note reading
      "This is a Hackathon with no provided dataset." All training data is externally sourced and
      must be disclosed, and submission is a write-up rather than a scored file.
- [x] Rule-by-rule audit written up in `docs/COMPLIANCE.md` — every published rule from the Kaggle
      competition rules, Kaggle's Foundational Rules, and the competition website, mapped to where
      it is enforced and how it is verified.
- [x] feat-003 — `uv run generate` writes a contract-valid 50,000-sequence library and ranked
      top 100 in about 2 seconds, byte-identical across processes.
- [x] feat-005 — Training corpus assembled and disclosed. `data/antibacterial.fasta` (39,448 AMPs,
      BSD-3, the organizers' own aggregation of dbAMP/DRAMP/DBAASP/CAMP/APD/+8) loaded via
      `src/amp_challenge_2027/data.py` with full metadata and a deterministic, valid, deduplicated
      `training_sequences()` accessor. Data card in `docs/DATA.md`; disclosed in `SUBMISSION.md`.
- [x] feat-006 — Generator ensemble trained. 8 AR-Transformers (10.68M params) one-per-H100 in
      ~135s. **Finding:** overfits fast on 37k seqs — val loss best at epoch 20 (~1.85), novelty
      falls from 0.95 (ep10) to 0.50 (ep80) as it memorizes. Early-stopping by val loss saves the
      right checkpoint: `best.pt` has novelty ~0.87, samples valid novel cationic/amphipathic
      peptides on CPU. Weights in `training/runs/` (gitignored).
- [x] feat-011 — Trained generator WIRED IN. `checkpoint/generator.pt` (gen6, 10.68M) ships; the
      submission now generates from the trained model, not the placeholder. Deterministic on
      GPU-if-available else CPU; ranks by model likelihood (interim until APEX). Full gate green:
      50k byte-identical across two GPU runs, 101 tests. Design driven by `docs/RESEARCH.md`
      (organizers' baseline = AMP-Diffusion + APEX; our model is CPU-capable, theirs isn't).
- [x] feat-008 — the organizers' own validator passes against the pushed public repo:
      *"All checks passed. Submission is valid!"* Re-run it after any change to the generator.

### What's In Progress

- [ ] feat-006 — Generative model behind `build_model()`
  - Train on `data.training_sequences()`. **Must generalize** beyond the corpus: it is also the
    novelty screen reference, so a memorizing model fails (see `docs/DATA.md`, "dual role").
  - Load weights from `checkpoint/` via `resolve_repo_path`; seed everything from `--seed`.
  - Two official starter kits target this exact format: `szczurek-lab/ampdiffusion-starter-kit`
    and `szczurek-lab/hydramp-starter-kit`.

### What's Next

1. feat-006 — put a real generative model behind `build_model()`.
2. feat-007 — implement a real `score()`; this is what the competition categories measure.
3. feat-009 — fill in the remaining `SUBMISSION.md` sections (abstract, model, ranking).
4. feat-008 — re-run the official verifier, then feat-010 — submit once (from `j_v_v_07`).

## Blockers / Risks

- [ ] **Two Kaggle accounts are reachable from this machine.** The competing account is
      `vijayavallabhj` / `j_v_v_07` (entry verified, `userHasEntered=True`). The *default* token at
      `~/.kaggle/kaggle.json` is a different account (`prakashchhipa`) and reports
      `userHasEntered=False` with a 403 on data. Run `kaggle config view` and confirm the username
      before trusting any API answer or submitting anything — Kaggle allows one account per
      participant. See `docs/COMPLIANCE.md` § Account identity.
- [ ] **Three days to the deadline.** The committed baseline is valid but scientifically empty.
      Treat it as a floor that guarantees a submittable entry, not as a candidate entry.
- [ ] **One entry per model.** There is no resubmission to fix a mistake, so feat-008 (the
      organizers' own validator, run against the pushed public URL) is mandatory before feat-010.
- [ ] **Top-100 novelty screen is the likely failure mode for a real model.** A model trained on
      known AMPs tends to reproduce them; anything above 80% Levenshtein identity to
      `data/antibacterial.fasta` is silently dropped and replaced by the next candidate. If a
      large fraction is rejected, the ranking is no longer the model's ranking.

## Decisions Made

- **uv only, no fallback path.** The organizers verify with `uv sync`; matching their toolchain
  exactly removes a class of environment failure. `uv.lock` and `.python-version` are committed.
- **Compliance encoded as a library module, not a script.** `constraints.py` is imported by the
  generator, so an invalid library fails at generation time with a non-zero exit rather than at
  submission time.
- **Reference set vendored into `data/`.** 4.3 MB, 39,448 sequences, from the official template.
  The generator needs it at run time to exclude exact matches and screen the top 100, and it
  makes the repo self-verifying.
- **`build_model()` as the single swap point.** Filtering, ranking, and writing stay fixed, so
  replacing the model cannot accidentally break the submission contract.
- **Baseline is labelled a placeholder in code and docs.** Uniform random peptides have no
  expected activity; the docstrings say so, to prevent a false baseline being read as a result.

## Files Modified This Session

- `src/amp_challenge_2027/constraints.py` — the competition's hard rules as code
- `src/amp_challenge_2027/data.py` — training corpus loading, metadata, disclosure (feat-005)
- `src/amp_challenge_2027/paths.py` — shared repo-root-aware path resolution
- `docs/DATA.md` — training data card
- `tests/test_data.py` — 21 tests over the data module
- `src/amp_challenge_2027/fasta.py` — FASTA I/O matching the official parser
- `src/amp_challenge_2027/model.py` — `PeptideGenerator` protocol and `RandomBaseline`
- `src/amp_challenge_2027/generate.py` — `uv run generate` entry point
- `tests/test_constraints.py` — 45 tests over the compliance layer
- `scripts/verify_submission.py` — vendored official validator (BSD-3, upstream attribution)
- `data/antibacterial.fasta` — vendored reference set
- `AGENTS.md`, `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh` — harness
- `README.md`, `docs/COMPETITION.md`, `SUBMISSION.md`, `LICENSE`

## Evidence of Completion

- [x] Tests pass: `uv run pytest -q` → `90 passed`
- [x] Generation: `uv run generate` → 50,000 records, 50,000 unique, lengths 8–50,
      0 alphabet violations, top.fasta 100 records all present in the library
- [x] Reproducibility: three fresh processes → library `6e24c32d5ff6b3988b7a1ef9401f5ab4`,
      top `ad0ac45cc621e865636e8bb70b607f24`
- [x] Official validator against the pushed public URL: `uv run python
      scripts/verify_submission.py https://github.com/Vijayavallabh/amp-challenge-2027`
      → "All checks passed. Submission is valid!" (7s, all 8 checks)

## Notes for Next Session

Start with `./init.sh`; it reproduces all of the evidence above in about ten seconds.

The interesting work is feat-006 and feat-007. Note that the two are scored differently: the
50,000-sequence library is screened computationally for diversity, novelty and physicochemical
distributions, while only 25 peptides drawn at random from the **top 50** are synthesized and
assayed. Optimising the library and optimising the top of the ranked list are not the same
problem, and the random draw means a single good sequence cannot carry the entry.

Two official starter kits target this exact submission format and are worth reading before
writing a model from scratch: `szczurek-lab/ampdiffusion-starter-kit` and
`szczurek-lab/hydramp-starter-kit`.

---

## Session 3 (2026-09-28) — optimize-to-deadline (goal: maximize win odds on 8xH100)

**Compute unlocked:** stood up a CUDA torch 2.5.1 env (scratch, gitignored) so APEX scores on
all 8 H100s: **50k peptides in 9.5s (~5.3k/s)**. Shipped path stays CPU-deterministic; GPU APEX is
offline only. Reusable `experiments/bulk_apex.py` (shards over CPU cores or 8 GPUs -> cached npz).

**Objective bake-off (experiments/analyze_objective.py, cached 50k lib):** tested a category-aligned
success-rate objective vs the shipped `broad_potency`. **Hypothesis DISPROVEN / not adopted:** on this
generator's pool, `broad_potency` already gives higher predicted Gram-neg SR (74 vs 64) and Broad SR
(57 vs 55); the category objective only trades GN -> GP/MDR. Keep `broad_potency`-style activity ranking.

**Real weaknesses found (predicted; no wet-lab claims):**
1. Pool is **Gram-negative-leaning**: top-50 predicted SR GN=74%, but **GP=28%**, MDR~35%. Gram-Positive
   coverage is the weak scored category. This is a *generator distribution* gap, not a selection gap.
2. Most APEX-active peptides are predicted **hemolytic** (P~0.76 pre-penalty) -> activity/selectivity tension.

**Plan (prioritized):**
- [feat-015] **ReST generator fine-tuning** on the 8 H100s: sample -> APEX+hemolysis score -> keep
  high-reward NOVEL, diverse samples -> low-LR fine-tune -> repeat. Reward = broad soft-Success-Rate
  (hard Gram+/MDR strains up-weighted) - lambda*P(hemolytic). Guards: novelty+diversity monitored every
  round (Phase-2 gate), physchem-envelope check, held-out APEX sub-model validation, saturating success
  (no chasing APEX's sub-uM tail). Goal: lift GP + selectivity without losing GN/broad. Fan out parallel
  chains (different lambda / GP-weight) across GPUs, pick best by held-out reward.
- [feat-016] **Deterministic parallel CPU-APEX** in shipped `oracle.py` (shard per-seq over cores;
  reassemble by sequence -> byte-identical). Cuts shipped runtime ~20x, affords larger oversample.
- Preserve the validated working submission at all times; re-run official validator before any ship.

**feat-015 ReST — validated + sweep running (2026-09-28):** 1-round smoke test: generator's
whole-pool active fraction **6.5% -> 50.5%**, novelty held 0.916 -> 0.934, charge 2.5 -> 6.0
(real-AMP-like), top-100 SR_broad 28.8 -> 38.5, GN 31.9 -> 46.3, P(hemolytic) 0.137 -> 0.071. No
collapse. Launched an 8-chain parallel Pareto sweep (one per H100) over lambda in {0.5,1,1.5,2} x
gp/mdr up-weight in {0,0.5,1}, 5 rounds each, plus a seed replicate. Diagnostics per round:
novelty (verbatim), div150 (diverse survivors of top-600 at 0.6 cap), physchem envelope, per-bucket
SR, P(hemolytic). Tools in experiments/ (bulk_apex, reward, sample_pool, rest_finetune,
preview_select, apex_permodel, pick_winner). Winner gets a held-out APEX sub-model check + a
reference-novelty check before it can replace checkpoint/generator.pt; then re-run the official
validator. Shipped deterministic path UNCHANGED until then.

**Sweep results + validation (2026-09-28, session 3):**
- 8-chain sweep: **gp_w drives the Gram balance** — gp=0.5 -> GN-specialist (GN~92, GP~26); gp=1.0 ->
  balanced (c3: GN 60/GP 53/MDR 50, phemo 0.046; c5: GN 67/GP 69/MDR 60, phemo 0.17). c1==c7 (seed
  replicate) -> ReST is reproducible. All chains lifted whole-pool active 6.5% -> ~79% over 5 rounds.
- **Anti-Goodhart checks PASS:** (a) top peptides are realistic amphipathic alpha-helical AMPs
  (e.g. ILGKLLSTAAKLLSKL, q=+3..+4, L15-18, breadth 9-10/11, minMIC ~1.5uM), not adversarial noise;
  (b) held-out APEX sub-model agreement 0.95 (>=6/8 submodels agree on 95% of active calls); breadth
  via submodels {0-3}=7.5 vs {4-7}=7.8, corr 0.67 -> activity generalizes across the ensemble, not
  gaming the mean; (c) physchem envelope sane; (d) verbatim novelty preserved (0.92 -> 0.97).
- **Only weakness: diversity collapse** — top-reward region concentrates to ~16 motif families
  (div150 150 -> ~16). Competition rewards a DIVERSE top-50 (25 drawn at random) + a diverse 50k
  library (Phase-2 screen), so this must be fixed. Launched a diversity-aware sweep (--dedup-cap 3,
  MinHash near-dup capping in the fine-tune set) from the diverse base generator: d0 (lam1,gp1),
  d1 (lam0.5,gp1), d2 (lam1,gp0.5), d3 (lam1.5,gp1).
- **feat-016 DONE + validated:** shipped CPU-APEX now shards over single-threaded workers ->
  byte-reproducible AND worker-count-independent (workers8==workers24, max|diff| 0.0), ~2x+ faster,
  memory-guarded auto worker count. 127 tests pass. Enables a larger shipped oversample.

**BREAKTHROUGH — sampling temperature fixes diversity AND novelty (2026-09-28):**
dedup-cap failed to stop collapse, and earlier rounds trade activity for diversity. But sampling
the ReST generator at **temperature 1.3** broadens the active manifold: on c3/round3, library
diversity 30% -> **57%**, novelty median 0.73 -> **0.60** (frac>0.8 0.27 -> 0.095), with top-50
**activity unchanged** (breadth 6.1, GN 71). So the shipped generator can be a strong late-ish ReST
round sampled hot -- active AND diverse AND novel. Decision converging on **c3/round3 + temperature
1.3**, category selection (gp_w 0.5, lambda 0.5, diversity 0.6). Shipped ranker + oracle already
updated (category_success_score); feat-016 parallel APEX makes the larger oversample affordable.
