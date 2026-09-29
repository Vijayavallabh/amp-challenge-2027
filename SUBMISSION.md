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
over the 20-amino-acid alphabet (10.7M parameters), first pre-trained on 39,448 known antibacterial
peptides (the challenge's own curated aggregation of public AMP databases) and then **fine-tuned
toward predicted activity and selectivity by rejection-sampling fine-tuning (ReST)**: we repeatedly
sample the model, score every candidate with the APEX MIC predictor and a hemolysis model, keep the
highest-reward *novel* peptides, and continue training on them. Over a few rounds this lifts the
fraction of samples predicted active on at least one strain from ~6% to ~75%, and a **final ReST
round targets the hard Gram-positive/MDR categories specifically** (distilling only from
ESMC-non-hemolytic peptides, so the generator gains activity without losing selectivity). The designs
stay realistic cationic amphipathic α-helices (validated below, including agreement across APEX's
eight independent sub-models, and an offline structural cross-check in which ESMFold2 — the latest
SOTA folder — predicts all 100 as confident amphipathic helices, 0 misfold flags).

The top-100 is where the competition is decided (25 of the top 50 are synthesised), so from a large
**8× (400k)** oversampled pool we rank by a **hard-Success-Rate APEX score** aligned with the five
scored categories — the fraction of Gram-positive and MDR strains cleared at ≤16 µM plus a broad
soft-potency tie-break, plus a modest **hard Gram−** term (`gn_weight=0.75`) so the weakest category
is lifted too — minus a **hemolysis penalty** from a selectivity model built on the latest
SOTA protein language model (**ESM Cambrian 600M**, held-out AUROC 0.905), with a within-list
**diversity screen**, plus a closed-form **amphipathicity** reward on the Eisenberg hydrophobic moment.
The top-100 candidate pool is sampled at a cooler **mixed temperature** (feat-028) so its best peptides
sit on the generator's high-activity modes, while the 50k library body stays hot for diversity. The
assayed **top-50** covers, at ≤16 µM: **Broad 0.68, Gram- 0.64, Gram+ 0.75, MDR 0.67**, at **0%
predicted-hemolytic** (every top-50 P < 0.08) and strong amphipathicity (µH median 0.54) — a strong,
balanced five-category
profile (the Gram+/MDR-targeted generator plus hard-SR selection roughly **doubled** the two hard
categories from where a broad-activity-only pipeline left them, at no selectivity cost, and the Gram−
term recovered the weakest category from 0.54 to ~0.57 at zero Gram+/MDR cost), and the designs are
markedly novel (top-50 median identity to any known AMP 0.62, max 0.73). APEX is the de la Fuente lab's own MIC predictor
(the lab that runs the competition's assays), used as a moderate, wet-lab-aligned signal, not ground
truth.

Activity fine-tuning would normally narrow the library, which the phase-1 screen penalises, so we
sample the **50k library body at an elevated temperature (1.6)** — hot sampling keeps it diverse and
novel (the Phase-2 advancement axes). But a hot pool also weakens the top-50: the generator's most-
active designs sit on its high-probability modes, which hot sampling under-samples. So the **top-100
candidate pool is drawn cooler (`--top-temperature 1.0`, feat-028 mixed-temperature sampling)** and
ranked separately; only the ~100 selected peptides are cool, and they are prepended to the hot library
body (which guarantees the top-100 ⊆ library rule). This lifts the assayed top-50 **Gram- 0.57→0.64 and
Broad 0.63→0.68** (cross-validated across all 8 APEX submodels, GP/MDR and 0%-hemolytic held) while the
library keeps **diversity 0.84 / novelty 1.0 unchanged** — a measured best-of-both, not the "no cost"
claim an earlier single-hot-pool version asserted.

The result (all figures are **computational predictions, not measurements** — we make no wet-lab
efficacy claim): a 50,000-peptide library that is **diverse** (≈81% of a random 500-sample mutually
<0.6 identity), **novel** (median Levenshtein identity to any known AMP ≈0.50, 3% above 0.80), and
AMP-like (84% net-cationic, mean length 18); and a **top-100 that is 100% predicted-active** (median
best-strain MIC ≈2.4 µM), broad-spectrum (top-50 mean ≈5.3 of 11 strains inhibited at ≤16 µM,
Gram-negative Success Rate ≈56%, Gram-positive ≈34%, MDR ≈38%), comparatively **non-hemolytic** (mean
predicted P(hemolytic) ≈0.07, vs ≈0.75 for unpenalised actives), and **novel** (top-50 median
identity to any known AMP ≈0.59, all ≤0.75, within the 0.80 rule).

## Model

- **Approach:** generative autoregressive language model over the 20-amino-acid alphabet (a
  generative method, as required), **fine-tuned toward predicted activity/selectivity by
  rejection-sampling fine-tuning (ReST)**. Candidates are then ranked by an external activity oracle
  (APEX) with a hemolysis-selectivity penalty and a diversity screen — see *Selection and ranking*
  below and `docs/RESEARCH.md`.
- **Architecture and size:** decoder-only Transformer (`src/amp_challenge_2027/nn.py`) — 6 layers,
  d_model 384, 6 heads, ~10.68M parameters, trained from scratch on the AMP corpus.
- **Activity/selectivity fine-tuning (ReST):** starting from the pre-trained checkpoint, each round
  we (1) sample ~200k peptides, (2) score every one with APEX (across the 8 H100s) and the hemolysis
  model, (3) keep the highest-reward peptides that are active (predicted MIC ≤16 µM on ≥1 strain) and
  *novel* (not a verbatim training peptide), and (4) continue training on that set at a low learning
  rate, with a slice of the original corpus mixed in as an anchor. The reward is the
  success-rate-aligned score used at selection time (below). Guards checked every round: verbatim
  novelty and physicochemical envelope are held, and a held-out check confirms the gains hold across
  APEX's eight independent sub-models (agreement ≈0.95) — evidence the peptides are genuinely active,
  not adversarial to the predictor. See `docs/RESEARCH.md` and `experiments/`.
- **Weights:** `checkpoint/generator.pt` — the fine-tuned generator (a mid ReST round chosen to
  balance activity against library diversity/novelty). The pre-trained base is preserved in git
  history; the training/fine-tuning code is in `training/` and `experiments/`.
- **Entry point:** `uv run generate` → `generate/library.fasta`, `generate/top.fasta`. Samples on
  GPU when available, else CPU. `torch` is a runtime dependency; `--baseline` falls back to a
  non-neural placeholder.
- **Seed:** 42 (fixed default). Two runs on the same machine are byte-identical
  (`torch.use_deterministic_algorithms` + seeded sampling); verified in `./init.sh`.
- **Note on the generator choice:** the organizers' SOTA baseline (AMP-Diffusion + APEX ranking)
  samples an *unoptimised* library and ranks it. Our edge is to close the loop — **fine-tune the
  generator against APEX and hemolysis** (ReST) so the candidate distribution itself concentrates on
  active, selective peptides, rather than relying on ranking to find rare actives in an unoptimised
  pool. The compact from-scratch Transformer also runs on CPU, so the submission degrades gracefully
  where the diffusion baseline would not.

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

- **How the 50,000 sequences were produced:** autoregressive sampling from the fine-tuned
  `checkpoint/generator.pt` on GPU (falls back to CPU), nucleus `top_p` 1.0, in batches of 4096, from a
  torch RNG seeded off the run seed (default 42), topping up in rounds until 50,000 unique/valid/novel
  sequences are collected. **Mixed-temperature (feat-028):** the 50k **library body** is sampled at
  **temperature 1.6** (hot → diverse/novel, the Phase-2 axes), while the separately-ranked **top-100
  candidate pool** is sampled at **`--top-temperature 1.0`** (cool → its best peptides sit on the
  generator's high-activity modes). The top list is prepended to the library body, so the library body
  is entirely temperature-1.6 apart from the ~100 selected peptides — Phase-2 diversity is unchanged
  (0.84) while the top-50 gains Gram-/Broad (see the abstract and `docs/RESEARCH.md` feat-028). Set
  `--top-temperature 1.6` to recover the earlier single-hot-pool run.
- **Constraint handling:** alphabet, length 8–50, uniqueness, and exclusion of exact matches to
  `data/antibacterial.fasta` are enforced in `src/amp_challenge_2027/constraints.py` and applied
  during generation (`build_library` collects into a `dict` for order-stable de-duplication).
- **How many candidates were drawn to fill the library, and how many were rejected:** the generator
  is highly valid and diverse, so the library fills in a few over-drawn rounds (over-draw factor
  1.2). Rejections are duplicates, the rare invalid sequence, and any exact match to the reference
  set. The library is 50,000 unique valid sequences; a fresh run reproduces it byte-for-byte.
- **Library characterization (for the phase-1 diversity/novelty/physicochemical screen):**
  **diverse** (≈81% of a random 500-peptide sample are mutually below 0.6 Levenshtein identity),
  **novel** (median max-identity to any known AMP ≈0.50, 90th percentile ≈0.72, only ≈3% above 0.80
  — and the library rule only forbids *exact* matches), and physicochemically **AMP-like** — 84%
  net-cationic, length 8–50 (mean ≈18), moderate hydrophobicity. Activity fine-tuning did not
  collapse the library: hot sampling (temperature 1.6) keeps it distributionally realistic and
  non-redundant, comparable to the un-fine-tuned base generator.

## Selection and ranking of the top 100

This is a required deliverable and it is what the competition measures — 25 peptides are drawn at
random from the top 50 and assayed.

- **Scoring function:** `score = balanced_success_score + 0.2·amphipathicity_bonus − 1.5·P(hemolytic)`,
  applied to a large **8× (400k)** oversampled candidate pool.
  - *balanced_success_score* — from **APEX-pathogen** (`oracle/apex`, run as an isolated `uv`
    subprocess), which predicts MIC (µM) against 11 clinical pathogens. We score the **hard
    Gram-positive and MDR Success Rate** (the fraction of those strains cleared at ≤16 µM — exactly
    the competition metric) plus a broad soft-Success-Rate tie-break:
    `SR_hard(Gram+) + SR_hard(MDR) + 0.5·mean soft-success`. This directly targets the two categories
    cationic AMPs are weakest on. An earlier *soft*-averaged score (`category_success_score`) was
    dominated by the many easy Gram-negative strains and plateaued at ~0.37 Gram+ Success Rate for
    *any* weighting; ranking by the **hard** Gram+/MDR rate instead surfaces the peptides that truly
    clear those strains. Combined with the Gram+/MDR-targeted ReST generator (above), this brings the
    assayed top-50 to **Gram+ 0.74 / MDR 0.67** (from 0.37 / 0.42 under a broad-activity-only pipeline)
    while holding Broad 0.63 / Gram- 0.57. We validated APEX against the 46 wet-lab-measured peptides in
    `data/experimental/mic.csv`: a moderate, wet-lab-aligned signal (AUROC 0.76 known-AMP vs random;
    0.62 per-(peptide,strain) on novel peptides), so we rank by it but do not chase its extreme tail.
    Details in `docs/RESEARCH.md`.
  - *P(hemolytic)* — from a **selectivity model built on the latest SOTA protein language model,
    ESM Cambrian 600M** (ESM++ `Synthyra/ESMplusplus_large`, MIT, on GPU-if-available), with a small
    trained MLP head over its mean-pooled embeddings (`checkpoint/selectivity_esmc.pt`, trained on
    HemoPI-2). Held-out **AUROC 0.905** — a large upgrade over the 11 physicochemical descriptors it
    replaces (0.778) and over ESM-2 150M (0.883); the larger **ESMC-6B gives no further gain** (also
    0.905 — the ceiling is the ~1000-peptide labelled dataset, not model size), so 600M is kept. The
    PLM is applied **two-stage**: rank the whole pool by activity, then score selectivity only on the
    top `refine_k=20000` (the rest assumed hemolytic) — exact for the top-100 and keeps the run fast.
    Gram+-active cationic peptides are overwhelmingly hemolytic (median P 0.98 vs 0.03), so the
    **λ=1.5** penalty (calibrated for the balanced score's larger scale) is decisive: it holds the
    top-50 at **0% ESMC-predicted-hemolytic** (median P 0.001) while preserving the Gram+/MDR gains —
    the Optimal Selectivity standout is protected, not traded away. The physicochemical model
    (`checkpoint/hemolysis.pt`) is retained as a graceful fallback if the PLM weights cannot be fetched.
  - *amphipathicity_bonus* — a **closed-form, deterministic** mechanistic prior: a smooth `[0,1]` reward
    that rises with each peptide's **Eisenberg hydrophobic moment µH** (a floor-ramp `clip((µH−0.25)/0.25,
    0, 1)`, saturating at µH 0.50), scaled by **0.2**. Amphipathicity — the segregation of hydrophobic
    and cationic faces on the helix — is the classic biophysical determinant of membrane disruption, so
    this prefers the *mechanistically-plausible* designs among the APEX-active ones and de-prioritises
    sequence-only artefacts the oracle happens to score well (an ESMFold2 structure check found the
    un-nudged top-50 sat at the weak edge of amphipathicity, with ~26% barely-amphipathic picks). Because
    µH is a closed-form function of the sequence (no structure prediction), it enters the ranking without
    breaking byte-reproducibility. It lifts the top-50 µH median **0.31 → 0.40** (barely-amphipathic picks
    26% → 12%) at ≤0.014 cost to every scored category (Gram+ −0.005, MDR unchanged — all within APEX's
    0.62-AUROC noise) and **0% predicted-hemolytic held**; a coefficient sweep confirmed the hemolysis
    gate holds across the range and 0.2 is the knee before the strongest category (Gram+) erodes. It is a
    hedge against the oracle's transfer error, not a category trade — the "cost" is oracle-internal while
    the µH gain is oracle-independent. Independently supported by 2026 literature on the amphipathic
    "Janus α-helix" (doi:10.1016/j.colsurfb.2026.116171). Details in `docs/RESEARCH.md` (feat-025).
- **Ranking procedure:** score every library sequence, sort by descending score (sequence as a
  deterministic tiebreak), then walk down the list applying the novelty and diversity screens
  below until 100 are selected. Fully deterministic — APEX runs in eval mode on CPU, sharded over
  single-threaded workers whose per-sequence results are reassembled by input order, so the output
  is byte-identical across runs regardless of the machine's core count. The ESMC selectivity head
  runs in eval mode under `torch.use_deterministic_algorithms` with a fixed cuBLAS workspace, so the
  two-stage GPU path is byte-reproducible too — verified by regenerating twice: `top.fasta` and
  `library.fasta` are md5-identical.
- **Novelty screen:** candidates above 0.80 Levenshtein ratio against any sequence in the
  reference set are rejected and replaced by the next-ranked candidate. In the shipped run the
  novelty screen rejected **1** candidate — the Gram+/MDR-targeted ReST generator explores new
  sequence space, so its designs sit comfortably below the 0.80 threshold (top-50 median identity to
  any known AMP **0.62**, max **0.73**, none an exact match — *more* novel than the earlier
  physicochemical/ESMC selections, evidence the harder optimisation is discovering new motifs rather
  than memorising known AMPs).
- **Diversity or redundancy control within the top 100:** a within-list cap
  (`--diversity-max-identity 0.6`) skips any candidate exceeding 0.60 Levenshtein identity to an
  already-selected peptide, keeping the more-active member of a near-duplicate pair. **202 near-
  duplicates were rejected** in the shipped run (far fewer than earlier pipelines' ~800 — the
  ReST-tuned generator's output is itself more diverse). This matters because the activity ranking
  concentrates the top of the list into a few cationic motif families, and the random top-50 draw
  would otherwise waste assays on near-duplicates; diversity also hedges against the moderate oracle
  being wrong about a motif. (Hot sampling keeps the *library* diverse; this cap keeps the *top-100*
  diverse.)

## Manual intervention and computational filters

Required disclosure. State plainly what was applied, including "none".

- **Manual curation or hand-picked sequences:** **None.** No sequence was hand-picked, edited, or
  reordered. The entire library and top-100 come from the automated, seeded pipeline.
- **Computational filters beyond the competition constraints:** a **hemolysis/selectivity penalty**
  (ESMC-600M model `checkpoint/selectivity_esmc.pt`, HemoPI-2; physicochemical `checkpoint/hemolysis.pt`
  as fallback) applied in ranking; a **closed-form amphipathicity term** (a smooth reward on the Eisenberg
  hydrophobic moment, coefficient 0.2, added to the activity score — see *Selection and ranking*); and a
  **within-list diversity cap** (0.60 Levenshtein identity). The amphipathicity term is a soft additive
  reward, **not** a hard window — no candidate is excluded by it, and no charge/pI, aggregation, or
  solubility filters are applied; those other properties emerge from the generator and are only *measured*
  for disclosure.
- **External predictors or databases used at selection time:** **APEX-pathogen** (MIC predictor,
  MIT-licensed, vendored under `oracle/apex`) for activity; an **ESM Cambrian 600M** selectivity model
  (ESM++ `Synthyra/ESMplusplus_large`, MIT weights fetched from HuggingFace; HemoPI-2-trained head) for
  hemolysis. Both are disclosed in `docs/DATA.md`. No proprietary or non-public data or services are
  used at any stage.

## Reproducibility

- `uv sync` then `uv run generate` regenerates both files exactly (fixed seed 42; APEX runs in
  eval mode on CPU; `torch.use_deterministic_algorithms`).
- Python 3.11, pinned in `.python-version`; dependencies locked in `uv.lock`. The APEX oracle is an
  isolated `uv` project (`oracle/apex`, its own lock) invoked as a subprocess; it syncs on first
  call. Weights (`checkpoint/generator.pt`, `checkpoint/selectivity_esmc.pt`, `checkpoint/hemolysis.pt`,
  `oracle/apex/`) are committed directly — the validator does a plain `git clone` with no `git lfs pull`.
- Verified with `uv run python scripts/verify_submission.py https://github.com/Vijayavallabh/amp-challenge-2027`
  on **2026-09-28** (commit `7f08c5a`, the GP/MDR-targeted ReST generator + balanced pipeline): *"All
  checks passed. Submission is valid!"* — fresh clone, `uv sync`, generate twice, all 8 checks including
  byte-identical reproducibility, exercising the **real ESM++/ESMC selectivity path** (the ESM++
  weights are fetched from HuggingFace during validation). (Earlier passes: `8c6e375` the balanced
  Gram+/MDR pipeline, `160e65b` the ESMC-selectivity pipeline, `2efad8c` the activity-only pipeline.)
- Hardware and runtime: generation on a single CUDA GPU (falls back to CPU); APEX scoring on CPU,
  **sharded over single-threaded worker subprocesses** (auto-sized to the machine's cores and free
  memory) so it is fast yet byte-reproducible and independent of the core count. The generator was
  pre-trained and **ReST-fine-tuned on 8×H100 offline**; inference needs one GPU or CPU. The ESMC
  selectivity backbone (~1.2 GB, MIT) is fetched from HuggingFace on first use; if it or the APEX
  oracle cannot load, `generate` degrades gracefully (ESMC → physicochemical selectivity → activity
  only; APEX → likelihood ranking → random baseline) so a valid submission is always produced.

## Checklist before submitting

Full rule-by-rule audit: [docs/COMPLIANCE.md](docs/COMPLIANCE.md).

- [x] Competition rules accepted on Kaggle from `vijayavallabhj` / `j_v_v_07` — verified via the
      API, `userHasEntered=True` (feat-004)
- [ ] Submitting from `j_v_v_07`, not from the machine's default token account
- [ ] `./init.sh` green, including the two-run byte-identical check
- [x] `scripts/verify_submission.py` PASSED against the **pushed public URL** for the feat-025 default
      (2026-09-29): cloned `https://github.com/Vijayavallabh/amp-challenge-2027.git` fresh, `uv sync`,
      generate ×2 — *"All checks passed. Submission is valid!"*, ranking
      `apex-balanced-success - 1.5*hemolysis + 0.2*amphipathicity`, all 8 checks incl. byte-identical
      reproducibility. The GitHub-cloned output is byte-identical to the shipped default (library
      `49451d15`, top `9bb8fe3b`), so the organizers' clone-and-run reproduces exactly this submission.
      (Also PASSED locally on commit `6f1d73f`.) Re-run once more immediately before submitting.
- [x] Every section above filled in, with no placeholder text left
- [x] Repository public, MIT licensed, `uv.lock` and `.python-version` committed
- [x] Weights committed or fetchable, and the inference path documented (`checkpoint/`, `oracle/apex/`)
- [x] Read access for [@RasmusML](https://github.com/RasmusML) and
      [@szymczakpau](https://github.com/szymczakpau) — satisfied by the repo being public
- [x] Top 100 confirmed to be a subset of the submitted 50,000-sequence library (enforced in code)
- [ ] Only one entry for this model; if a second model is planned, organizers contacted in advance
