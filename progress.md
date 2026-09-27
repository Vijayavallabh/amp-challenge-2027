# Session Progress Log

## Current State

**Last Updated:** 2026-09-27
**Active Feature:** feat-013 — APEX activity oracle for top-100 ranking (the win lever)
**Deadline:** 2026-09-30 22:00 UTC (1 October 2026, AOE) — see `docs/COMPETITION.md`

The repository is initialized and produces a structurally valid, reproducible submission. The
model behind it is a placeholder with no expected antimicrobial activity, so the scientific work
has not started.

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
