# AGENTS.md

Submission repository for the **AMP Challenge 2027** (NeurIPS Competition Track) — generative
design of antimicrobial peptides. The organizers clone this repo, run `uv sync`, then
`uv run generate`, and validate the output. Everything here exists to keep that path working.

Competition facts (deadline, deliverables, evaluation, bacterial panel) live in
[docs/COMPETITION.md](docs/COMPETITION.md). The rule-by-rule audit — every published rule, where
it is enforced, how it is verified — is [docs/COMPLIANCE.md](docs/COMPLIANCE.md). Usage lives in
[README.md](README.md). This file is routing and invariants only.

**Submit only from the Kaggle account `j_v_v_07`.** The API token on this machine belongs to a
different account; using it would breach Kaggle's one-account rule. See
[docs/COMPLIANCE.md](docs/COMPLIANCE.md) § Account identity.

## Startup Workflow

1. `pwd` — confirm you are in the repo root. Relative paths (`data/`, `checkpoint/`) assume it.
2. Read this file, then `docs/COMPETITION.md` for the rules you are being scored against.
3. Run `./init.sh` — installs, tests, generates, and checks reproducibility.
4. Read `feature_list.json` for feature state and `progress.md` for where the last session left off.
5. `git log --oneline -5`.

If `./init.sh` fails, fix that before adding scope. A red baseline means there is no submission.

## Invariants

Break any of these and the submission is rejected, not merely scored badly.

1. **`uv run generate` is the entry point.** It must succeed with no arguments and write
   `generate/library.fasta` (exactly 50,000 sequences) and `generate/top.fasta` (exactly 100,
   all of them present in the library). Every flag keeps a default.
2. **Output is byte-identical across runs.** The validator generates twice and compares bytes.
   Seed every RNG from `--seed` (default 42).
3. **uv is the only environment tool.** `uv add` / `uv remove`, never `pip install`, conda,
   poetry, or hand-edited `[project.dependencies]`. `uv.lock` and `.python-version` stay committed.
4. **Sequence constraints are absolute** — 20 standard residues (`ACDEFGHIKLMNPQRSTVWY`), length
   8–50, no duplicates, linear with free termini. The library must contain nothing identical to
   `data/antibacterial.fasta`; the top 100 must stay at or below 80% Levenshtein identity to
   every sequence in it. These are encoded in `src/amp_challenge_2027/constraints.py` — that
   module is the contract. Change it only to track an upstream change in
   `scripts/verify_submission.py`, and update the tests in the same commit.
5. **The repo stays public and MIT-licensed**, and any training data must be publicly
   releasable under a permissive license. Both are conditions of co-authorship eligibility.

## Reproducibility Traps

These have all bitten real submissions. Reproducibility is checked, so they are correctness bugs.

- **Never iterate a `set` to produce output.** Python randomizes string hashing per process, so
  set order differs between the validator's two runs. Use a list, or a `dict` for ordered
  de-duplication — that is why `build_library` collects into a `dict`.
- **Sort with an explicit tiebreaker.** `sorted(..., key=lambda p: (-score, seq))`, not bare
  `reverse=True`, so equal scores cannot reorder.
- **Resolve data and checkpoint paths through `resolve_repo_path`.** The validator sets the repo
  root as the working directory; a bare relative `open()` breaks anywhere else.
- **Seed every source of randomness**, including any framework RNG (`torch.manual_seed`,
  `random.seed`) and non-deterministic GPU kernels, not just numpy.

## Working Rules

- One feature at a time from `feature_list.json`. Mark it `in-progress` before starting.
- **Stay in scope.** Modelling work touches `model.py` and `build_model()` in `generate.py`, and
  nothing else. Don't refactor the compliance layer, the I/O, or the harness while adding a model.
- Don't commit `generate/` output — it is regenerated and would bloat the repo.
- Don't weaken a constraint or a test to make a run pass. If the generator cannot fill 100 novel
  candidates, that is a finding about the model, not about the screen.

### Scope boundaries

| Area | Rule |
|---|---|
| `src/amp_challenge_2027/model.py`, `build_model()` | Where model work belongs. Edit freely. |
| `src/amp_challenge_2027/nn.py` | The generator architecture, shared by training and the shipped `generate`. Change with care — a change here must keep `checkpoint/generator.pt` loadable, or ship a new checkpoint. |
| `checkpoint/generator.pt` | Shipped trained weights (committed, ~41MB). Replace via `training/` when a better model is selected; keep `generate` byte-reproducible. |
| `src/amp_challenge_2027/data.py` | Training corpus loading and disclosure. Extend for new data sources; keep `docs/DATA.md` in step. |
| `src/amp_challenge_2027/constraints.py` | The submission contract. Change only to track an upstream change in `scripts/verify_submission.py`, with tests in the same commit. |
| `src/amp_challenge_2027/generate.py` (outside `build_model`) | Filtering, ranking, and writing. Leave alone unless the feature is explicitly about them. |
| `scripts/verify_submission.py`, `data/antibacterial.fasta` | Vendored upstream, BSD-3. Never edit; re-vendor if upstream changes. |
| `LICENSE`, `pyproject.toml` licence and entry point | Fixed by the competition rules. Changing them can disqualify the entry. |
| `generate/` | Build output. Never commit. |

## Definition of Done

A feature is done only when all of these hold:

- [ ] Behavior implemented and the model reachable through `build_model()`
- [ ] `./init.sh` passes, including the two-run byte-identical check
- [ ] Tests added or updated for the change, and `uv run pytest -q` is green
- [ ] Evidence (command + result) recorded in `feature_list.json` and `progress.md`
- [ ] Clean checkout can run `./init.sh` immediately

## End of Session

1. Update `progress.md` (state, evidence, next action) and `feature_list.json` (status).
2. Record unresolved risks and blockers.
3. Commit with a descriptive message; push so the public repo reflects reality.
4. Leave the repo runnable from `./init.sh` with no manual setup.

## Escalation

- **Rule ambiguity** — check `docs/COMPETITION.md`, then the official FAQ or a GitHub issue on
  `szczurek-lab/amp-challenge-2027`. Do not guess at a rule that decides acceptance.
- **A constraint that cannot be satisfied** — record it in `progress.md` and flag it; do not
  relax `constraints.py`.
- **Anything touching what gets submitted or published** — surface to the user rather than
  deciding. Submission is one-shot per model.
