# Submission write-up

Four things are required deliverables: the abstract, the training-data summary, the selection and
ranking procedure, and disclosure of any manual intervention or filters. Fill each section in
before submitting — see `docs/COMPETITION.md` for the exact wording of the requirements.

**Status: not yet written.** The repository currently generates from a uniform-random placeholder.

---

## Team

- **Team name:** _TBD_
- **Members and affiliations:** Vijayavallabh (IIT Madras)
- **Kaggle display name:** _TBD — this is what the CC BY 4.0 write-up is attributed to_
- **Contact:** be23b041@smail.iitm.ac.in
- **Repository:** https://github.com/Vijayavallabh/amp-challenge-2027

## Abstract

_A summary of the method. What kind of generative model, what it was trained on, what makes the
designs novel, and how candidates were prioritized._

## Model

- **Approach:** _e.g. language model, diffusion, VAE, GFlowNet, RL, Bayesian optimization,
  evolutionary, or a hybrid. Only generative methods are permitted._
- **Architecture and size:**
- **Conditioning or guidance:**
- **Weights:** _path under `checkpoint/`, and how they were produced_
- **Entry point:** `uv run generate` → `generate/library.fasta`, `generate/top.fasta`
- **Seed:** 42 (fixed default; two runs are byte-identical)

## Training data

Every source must be publicly redistributable under a permissive licence, or released as part of
the submission. Anything non-public that cannot be released costs co-authorship eligibility.

| Source | Version / accessed | Records used | Licence | Redistributable |
|---|---|---|---|---|
| _e.g. DBAASP_ | | | | |
| _e.g. APD3_ | | | | |
| _e.g. dbAMP_ | | | | |
| _e.g. Peptipedia_ | | | | |

- **Preprocessing:** _de-duplication, length and alphabet filtering, clustering, splits_
- **Held-out data:** _what was excluded from training and why_
- **Non-public data:** _none, or state what is being released and under which licence_

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

- [ ] Competition rules accepted on Kaggle (feat-004)
- [ ] `./init.sh` green, including the two-run byte-identical check
- [ ] `scripts/verify_submission.py` run against the **pushed public URL** and passing
- [ ] Every section above filled in, with no placeholder text left
- [ ] Repository public, MIT licensed, `uv.lock` and `.python-version` committed
- [ ] Weights committed or fetchable, and the inference path documented
- [ ] Read access granted to [@RasmusML](https://github.com/RasmusML) and
      [@szymczakpau](https://github.com/szymczakpau)
- [ ] Top 100 confirmed to be a subset of the submitted 50,000-sequence library
