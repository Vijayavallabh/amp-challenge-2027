# Submission write-up

Four things are required deliverables: the abstract, the training-data summary, the selection and
ranking procedure, and disclosure of any manual intervention or filters. Fill each section in
before submitting — see `docs/COMPETITION.md` for the exact wording of the requirements.

**Status: not yet written.** The repository currently generates from a uniform-random placeholder.

---

## Team

- **Team name:** _TBD_
- **Members and affiliations:** Vijayavallabh (IIT Madras)
- **Kaggle account:** `vijayavallabhj`, display name **`j_v_v_07`** — entry confirmed
  (`userHasEntered=True`, verified 2026-09-27). Submit from this account and no other. The write-up
  is licensed CC BY 4.0 and attributed to the display name; to change that later, contact
  support@kaggle.com.
- **Contact:** be23b041@smail.iitm.ac.in
- **Repository:** https://github.com/Vijayavallabh/amp-challenge-2027 (public, MIT)

> The *default* Kaggle token on this machine (`~/.kaggle/kaggle.json`) belongs to a different
> account (`prakashchhipa`). Never submit with it — check `kaggle config view` first. See
> `docs/COMPLIANCE.md` § Account identity.

## Abstract

_A summary of the method. What kind of generative model, what it was trained on, what makes the
designs novel, and how candidates were prioritized._

## Model

- **Approach:** _e.g. language model, diffusion, VAE, GFlowNet, RL, Bayesian optimization,
  evolutionary, or a hybrid. **Only generative methods are permitted** — state plainly which
  generative method this is._
- **Architecture and size:**
- **Conditioning or guidance:**
- **Weights:** _path under `checkpoint/`, and how they were produced_
- **Entry point:** `uv run generate` → `generate/library.fasta`, `generate/top.fasta`
- **Seed:** 42 (fixed default; two runs are byte-identical)

## Training data

The competition provides **no dataset** — its only data file is a note saying so — so all training
data is externally sourced and disclosed here. Full data card: [docs/DATA.md](docs/DATA.md).

**Corpus:** `data/antibacterial.fasta` — 39,448 antimicrobial peptides, vendored verbatim from the
official challenge template ([szczurek-lab/amp-challenge-2027](https://github.com/szczurek-lab/amp-challenge-2027))
and used as-is. It is the organizers' own curated aggregation of public AMP databases, already
pre-filtered to the competition constraints (20 standard residues, length 8–50, unique).

| Source | Provenance | Records | Licence | Redistributable |
|---|---|---:|---|---|
| `antibacterial.fasta` (aggregation) | Official template, MD5 `8366eb2c…` | 39,448 | BSD-3-Clause | **Yes** — already public in the template repo |
| ↳ aggregates: dbAMP, DRAMP, DBAASP, CAMP, SATPdb, APD, +7 | public AMP databases, open for research | — | per source | via the BSD-3 aggregation above |

- **Preprocessing:** validity filter (20 AA, length 8–50) and de-duplication in
  `data.training_sequences()`; both are no-ops on this corpus (already 100% valid and unique) and
  exist as safety nets. No manual curation, no hand-picking, no external label used to select data.
- **Held-out data:** none held out at present. Note the corpus doubles as the novelty screen
  reference, so the model must generalize beyond it (see the data card, "dual role").
- **Non-public data:** none. Nothing proprietary or non-redistributable is used.
- **Why not more sources:** deferred deliberately — each raw database has its own terms of use, they
  overlap heavily with this aggregation, and the vendored set already covers the endorsed sources
  under a clean permissive licence. See the data card for the full rationale.

## Library generation

- **How the 50,000 sequences were produced:** _sampling procedure, temperature, batching_
- **Constraint handling:** alphabet, length 8–50, uniqueness, and exclusion of exact matches to
  `data/antibacterial.fasta` are enforced in `src/amp_challenge_2027/constraints.py` and applied
  during generation.
- **How many candidates were drawn to fill the library, and how many were rejected:**

## Selection and ranking of the top 100

This is a required deliverable and it is what the competition measures — 25 peptides are drawn at
random from the top 50 and assayed.

- **Scoring function:** _what `score()` predicts and how it was fitted or calibrated_
- **Ranking procedure:**
- **Novelty screen:** candidates above 0.80 Levenshtein ratio against any sequence in the
  reference set are rejected and replaced by the next-ranked candidate.
  **Report how many were rejected** — a large number means the submitted ranking has drifted from
  the model's own ranking.
- **Diversity or redundancy control within the top 100:**

## Manual intervention and computational filters

Required disclosure. State plainly what was applied, including "none".

- **Manual curation or hand-picked sequences:**
- **Computational filters beyond the competition constraints:** _e.g. hemolysis or toxicity
  predictors, charge or hydrophobicity windows, aggregation or solubility filters_
- **External predictors or databases used at selection time:**

## Reproducibility

- `uv sync` then `uv run generate` regenerates both files exactly.
- Python 3.11, pinned in `.python-version`; dependencies locked in `uv.lock`.
- Verified with `uv run python scripts/verify_submission.py <repo-url>` on _TBD (date, result)_.
- Hardware and runtime: _TBD_

## Checklist before submitting

Full rule-by-rule audit: [docs/COMPLIANCE.md](docs/COMPLIANCE.md).

- [x] Competition rules accepted on Kaggle from `vijayavallabhj` / `j_v_v_07` — verified via the
      API, `userHasEntered=True` (feat-004)
- [ ] Submitting from `j_v_v_07`, not from the machine's default token account
- [ ] `./init.sh` green, including the two-run byte-identical check
- [ ] `scripts/verify_submission.py` run against the **pushed public URL** and passing
- [ ] Every section above filled in, with no placeholder text left
- [x] Repository public, MIT licensed, `uv.lock` and `.python-version` committed
- [ ] Weights committed or fetchable, and the inference path documented
- [x] Read access for [@RasmusML](https://github.com/RasmusML) and
      [@szymczakpau](https://github.com/szymczakpau) — satisfied by the repo being public
- [x] Top 100 confirmed to be a subset of the submitted 50,000-sequence library (enforced in code)
- [ ] Only one entry for this model; if a second model is planned, organizers contacted in advance
