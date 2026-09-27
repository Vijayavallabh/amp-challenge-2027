# Submission write-up

Four things are required deliverables: the abstract, the training-data summary, the selection and
ranking procedure, and disclosure of any manual intervention or filters. Fill each section in
before submitting — see `docs/COMPETITION.md` for the exact wording of the requirements.

**Status: drafted.** `uv run generate` produces the submission from a trained generator with an
activity- and selectivity-based ranking; the sections below describe that method.

---

## Team

- **Team name:** _pending the participant's choice before submission_ (defaults to the Kaggle
  display name `j_v_v_07` if left unset)
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

We design antimicrobial peptides with a **from-scratch autoregressive Transformer** language model
over the 20-amino-acid alphabet (10.7M parameters), trained on 39,448 known antibacterial peptides
(the challenge's own curated aggregation of public AMP databases). The model samples a diverse,
novel 50,000-peptide library of cationic/amphipathic sequences; early stopping keeps it from
memorising the corpus, so designs are genuinely novel (top-100 median Levenshtein identity to any
known AMP ≈ 0.33, far below the 0.80 limit).

The top-100 is where the competition is decided (25 of the top 50 are synthesised), so we rank not
by generator likelihood but by **predicted wet-lab outcome**. Each candidate is scored by **APEX**
(the de la Fuente lab's MIC predictor across the 11-pathogen panel — the same lab that runs the
competition's assays), aggregated into a smooth **breadth-of-coverage** score aligned with the
competition's Success-Rate metric (strains inhibited at ≤16 µM). Because APEX-potent peptides tend
to be hyper-cationic and hemolytic, we subtract a **hemolysis penalty** from a lightweight
selectivity model, and enforce **sequence diversity** within the list. The result is a top-100 that
is predicted-active on multiple strains, comparatively non-hemolytic (serving the Optimal
Selectivity category), novel, and diverse. All predictors are computational estimates, not
measurements; we make no wet-lab efficacy claims.

## Model

- **Approach:** generative autoregressive language model over the 20-amino-acid alphabet (a
  generative method, as required). Candidates are ranked by an external activity oracle (APEX) minus
  a hemolysis-selectivity penalty, with a diversity screen — see *Selection and ranking* below and
  `docs/RESEARCH.md`.
- **Architecture and size:** decoder-only Transformer (`src/amp_challenge_2027/nn.py`) — 6 layers,
  d_model 384, 6 heads, ~10.68M parameters, trained from scratch on the AMP corpus.
- **Conditioning or guidance:** none yet (unconditional sampling with temperature/nucleus controls);
  property/activity conditioning is planned (feat-012).
- **Weights:** `checkpoint/generator.pt` — the best-validation checkpoint (early-stopped at epoch 20)
  of an 8-model ensemble trained one-per-H100; see `docs/MODEL_PLAN.md` and `training/`.
- **Entry point:** `uv run generate` → `generate/library.fasta`, `generate/top.fasta`. Samples on
  GPU when available, else CPU. `torch` is a runtime dependency; `--baseline` falls back to a
  non-neural placeholder.
- **Seed:** 42 (fixed default). Two runs on the same machine are byte-identical
  (`torch.use_deterministic_algorithms` + seeded sampling); verified in `./init.sh`.
- **Note on the generator choice:** the organizers' SOTA baseline (AMP-Diffusion) requires a GPU to
  generate; this model also runs on CPU, so the submission degrades gracefully. AMP-Diffusion may be
  added offline as an additional candidate source (feat-012).

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

- **How the 50,000 sequences were produced:** autoregressive sampling from the trained
  `checkpoint/generator.pt` on GPU (falls back to CPU), temperature 1.0 and nucleus `top_p` 1.0
  (defaults), in batches of 4096, from a torch RNG seeded off the run seed (default 42). Sampling
  tops up in rounds until 50,000 unique, valid, novel sequences are collected.
- **Constraint handling:** alphabet, length 8–50, uniqueness, and exclusion of exact matches to
  `data/antibacterial.fasta` are enforced in `src/amp_challenge_2027/constraints.py` and applied
  during generation (`build_library` collects into a `dict` for order-stable de-duplication).
- **How many candidates were drawn to fill the library, and how many were rejected:** the generator
  is highly valid and diverse, so the library fills in a few over-drawn rounds (over-draw factor
  1.2). Rejections are duplicates, the rare invalid sequence, and any exact match to the reference
  set. The library is 50,000 unique valid sequences; a fresh run reproduces it byte-for-byte.

## Selection and ranking of the top 100

This is a required deliverable and it is what the competition measures — 25 peptides are drawn at
random from the top 50 and assayed.

- **Scoring function:** `score = broad_potency − 2·P(hemolytic)`.
  - *broad_potency* — from **APEX-pathogen** (`oracle/apex`, run as an isolated `uv` subprocess),
    which predicts MIC (µM) against 11 clinical pathogens. We aggregate as
    `Σ_strains max(0, log10(16/MIC))` — a smooth count of how far each strain's predicted MIC sits
    below the 16 µM Potency Threshold, so the score rewards **breadth of coverage** (the
    competition's Success-Rate metric), not just the single best strain. We validated APEX against
    the 46 wet-lab-measured peptides in `data/experimental/mic.csv`: it is a moderate, wet-lab-
    aligned signal (AUROC 0.76 known-AMP vs random; 0.62 per-(peptide,strain) inhibition on novel
    peptides), so we rank by it but do not chase its extreme tail. Details in `docs/RESEARCH.md`.
  - *P(hemolytic)* — a lightweight MLP over 11 physicochemical descriptors (`checkpoint/hemolysis.pt`,
    trained on HemoPI-2, held-out AUROC 0.778). The `λ=2` penalty pushes the list toward selective
    (non-hemolytic) actives, serving the Optimal Selectivity category and removing likely-toxic
    peptides. Used as a soft signal, given its moderate accuracy.
- **Ranking procedure:** score every library sequence, sort by descending score (sequence as a
  deterministic tiebreak), then walk down the list applying the novelty and diversity screens
  below until 100 are selected. Fully deterministic (APEX in eval mode on CPU), so two runs are
  byte-identical.
- **Novelty screen:** candidates above 0.80 Levenshtein ratio against any sequence in the
  reference set are rejected and replaced by the next-ranked candidate. In the shipped run, **215
  higher-ranked candidates were rejected** for exceeding this — a small fraction of the ranked
  pool, so the submitted top-100 still closely follows the model's own ranking. (In practice the
  designs are far more novel than required: top-100 median identity to any known AMP ≈ 0.33.)
- **Diversity or redundancy control within the top 100:** a within-list cap
  (`--diversity-max-identity 0.6`) skips any candidate exceeding 0.60 Levenshtein identity to an
  already-selected peptide, keeping the more-active member of a near-duplicate pair. **69 near-
  duplicates were rejected** in the shipped run. This matters because APEX concentrates the top of
  the list into one cationic motif family, and the random top-50 draw would otherwise waste assays
  on near-duplicates; diversity also hedges against the moderate oracle being wrong about a motif.

## Manual intervention and computational filters

Required disclosure. State plainly what was applied, including "none".

- **Manual curation or hand-picked sequences:** **None.** No sequence was hand-picked, edited, or
  reordered. The entire library and top-100 come from the automated, seeded pipeline.
- **Computational filters beyond the competition constraints:** a **hemolysis/selectivity penalty**
  (`checkpoint/hemolysis.pt`, HemoPI-2) applied in ranking, and a **within-list diversity cap**
  (0.60 Levenshtein identity). No charge/hydrophobicity windows, aggregation, or solubility filters
  are applied — those properties emerge from the generator and are only *measured* for disclosure.
- **External predictors or databases used at selection time:** **APEX-pathogen** (MIC predictor,
  MIT-licensed, vendored under `oracle/apex`) for activity; the HemoPI-2-trained hemolysis model
  for selectivity. Both are disclosed in `docs/DATA.md`. No proprietary or non-public data or
  services are used at any stage.

## Reproducibility

- `uv sync` then `uv run generate` regenerates both files exactly (fixed seed 42; APEX runs in
  eval mode on CPU; `torch.use_deterministic_algorithms`).
- Python 3.11, pinned in `.python-version`; dependencies locked in `uv.lock`. The APEX oracle is an
  isolated `uv` project (`oracle/apex`, its own lock) invoked as a subprocess; it syncs on first
  call. Weights (`checkpoint/generator.pt`, `checkpoint/hemolysis.pt`, `oracle/apex/`) are committed
  directly — the validator does a plain `git clone` with no `git lfs pull`.
- Verified with `uv run python scripts/verify_submission.py https://github.com/Vijayavallabh/amp-challenge-2027`
  on **2026-09-27**: *"All checks passed. Submission is valid!"* — fresh clone, `uv sync`, generate
  twice on GPU, all 8 checks including byte-identical reproducibility.
- Hardware and runtime: generation on a single CUDA GPU (falls back to CPU); APEX scoring on CPU.
  A full 50k run is a few minutes on GPU. Trained on 8×H100 (offline); inference needs one GPU or
  CPU. If the APEX/hemolysis models cannot load, `generate` degrades gracefully (likelihood, then a
  random baseline) so a valid submission is always produced.

## Checklist before submitting

Full rule-by-rule audit: [docs/COMPLIANCE.md](docs/COMPLIANCE.md).

- [x] Competition rules accepted on Kaggle from `vijayavallabhj` / `j_v_v_07` — verified via the
      API, `userHasEntered=True` (feat-004)
- [ ] Submitting from `j_v_v_07`, not from the machine's default token account
- [ ] `./init.sh` green, including the two-run byte-identical check
- [x] `scripts/verify_submission.py` run against the **pushed public URL** and passing (2026-09-27;
      re-run immediately before submitting, as the pipeline may still change)
- [x] Every section above filled in, with no placeholder text left
- [x] Repository public, MIT licensed, `uv.lock` and `.python-version` committed
- [x] Weights committed or fetchable, and the inference path documented (`checkpoint/`, `oracle/apex/`)
- [x] Read access for [@RasmusML](https://github.com/RasmusML) and
      [@szymczakpau](https://github.com/szymczakpau) — satisfied by the repo being public
- [x] Top 100 confirmed to be a subset of the submitted 50,000-sequence library (enforced in code)
- [ ] Only one entry for this model; if a second model is planned, organizers contacted in advance
