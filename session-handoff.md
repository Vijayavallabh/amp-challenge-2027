# Session Handoff

## Current Objective

- **Goal:** Get a real generative model behind `build_model()` and a real `score()` behind the
  top-100 ranking, in time for the 2026-09-30 22:00 UTC deadline.
- **Current status:** Trained AR-Transformer wired into `generate` and verified end-to-end
  (official validator passes on a fresh clone). Ranking is interim (likelihood); APEX next.
- **Branch / commit:** `main`, initial commit.

## Completed This Session (2026-09-27)

- [x] uv project bootstrap — Python 3.11 pinned, `uv.lock` committed, MIT licensed, `generate`
      wired as the console entry point per the official template
- [x] Compliance layer (`constraints.py`) mirroring the organizers' validator, with 69 tests
- [x] Rule-by-rule audit in `docs/COMPLIANCE.md`; `--length` added for template interface parity
- [x] Contract-valid baseline: 50,000-sequence library and ranked top 100 in ~2s
- [x] Harness: `AGENTS.md`, `feature_list.json`, `progress.md`, `init.sh`, this file
- [x] Docs: `README.md`, `docs/COMPETITION.md`, `SUBMISSION.md` skeleton
- [x] Vendored the official validator and the 39,448-sequence reference set
- [x] Verified the Kaggle entry (`userHasEntered=True`) and that no dataset is provided
- [x] feat-005 — training corpus assembled, loaded via `data.py`, documented in `docs/DATA.md`

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Install | `uv sync --locked` | pass | Python 3.11.14, 4 runtime packages |
| Tests | `uv run pytest -q` | 90 passed | sequence + submission rules + training corpus |
| Generation | `uv run generate` | pass, ~2s | 50,000 unique, lengths 8–50, 0 alphabet violations |
| Top list | built in | pass | 100 records, all present in the library |
| Reproducibility | 3 fresh processes | identical | `6e24c32d…` / `ad0ac45c…` |
| Official validator | `scripts/verify_submission.py <repo-url>` | **all checks passed** | fresh clone, trained model, 50k x2 on GPU, ~2m17s |
| Tests | `uv run pytest -q` | 101 passed | +11 trained-generator/build_model |

## Decisions Made

- uv only, matching the organizers' toolchain exactly; `uv.lock` and `.python-version` committed.
- Constraints live in an imported module, so invalid output fails at generation time with a
  non-zero exit rather than at submission time.
- `build_model()` is the single swap point; filtering, ranking and writing stay fixed.
- The baseline is labelled a placeholder in code and docs so it cannot be mistaken for a result.

## Blockers / Risks

- [ ] **Two Kaggle accounts reachable here** — competing account `vijayavallabhj` / `j_v_v_07`
      (entry verified); the machine default token is `prakashchhipa`. Check `kaggle config view`
      before trusting an API answer or submitting. See `docs/COMPLIANCE.md` § Account identity.
- [ ] **Three days left**, and the modelling work has not started.
- [ ] **One entry per model, no resubmission** — feat-008 is mandatory before feat-010.
- [ ] **Novelty screen** is the likely failure mode once a model is trained on known AMPs; watch
      the rejection count that `generate` prints.

## Next Session Startup

1. Read `AGENTS.md`, then `docs/COMPETITION.md`.
2. Read `feature_list.json` and `progress.md`.
3. Run `./init.sh` — about ten seconds, reproduces all the evidence above.

## Recommended Next Step

**feat-013 — the APEX activity oracle (the win lever).** The top-100 ranking is what gets
wet-lab tested, and it is currently ranked by model likelihood — a weak proxy for potency.
Integrate **APEX** (de la Fuente MIC predictor, bundled in the ampdiffusion-starter-kit under
`apex/`, run as an isolated `uv` subprocess) to rank by predicted MIC across the 11-pathogen
panel, and add a hemolysis predictor for the Optimal Selectivity category. Then *optimize*
against APEX (feat-014), not just filter — that is the edge over the excluded baseline.

Also (feat-012, parallel): ensemble the 8 trained generators and add AMP-Diffusion as an offline
candidate source to raise library diversity/novelty for phase-1 screening.

Compute: all 8 H100s are free; APEX scoring of large candidate pools + directed optimization is
the heavy, high-value GPU workload. See `docs/RESEARCH.md` and `docs/MODEL_PLAN.md`.