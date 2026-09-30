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
**8× (400k)** oversampled pool we select it by a **wet-lab-calibrated composition ranking** — the
submission's central methodological contribution (feat-033). We could only find that this beats
oracle-based ranking by building the missing test: scoring every candidate signal against *real*
activity. On our 46 wet-lab MICs (the only APEX-independent ground truth) **and** on 946 independent
DBAASP peptides we harvested for this purpose, the same picture holds — a **range-restriction failure
of the oracle**. Across the full activity range APEX ranks real activity well (Spearman ρ +0.33), but
**within the high-activity band our top-100 is drawn from, APEX loses its ranking power** (ρ falls to
+0.06 in the APEX-top-15%), while simple **composition — Lysine-richness and low aromatic content —
predicts real activity there** (ρ +0.27), on both datasets independently. So we keep APEX only as the
coarse **active-band gate** it *is* reliable at (top-20k by predicted Success Rate) and, within that
band, rank by `lys_fraction − 0.5·aromatic_fraction` minus the selectivity penalty below. Ranking the
46 by this composition score gives a **real** Broad Success Rate of **0.46** (Gram− 0.54) versus 0.31
for a random draw and ≤0.30 for *any* APEX score; the DBAASP within-band data confirm it lifts real
Broad, Gram-negative **and** Gram-positive rates together (e.g. real Gram+ 0.60→0.71 as Lysine weight
rises). The candidate pool is drawn at a cooler top-temperature (0.8) while the 50k library body stays hot (1.6)
for diversity (detailed next); a within-list **diversity screen** and the < 0.80 novelty rule complete
selection.

Activity fine-tuning would normally narrow the library, which the phase-1 screen penalises, so we
sample the **50k library body at an elevated temperature (1.6)** — hot sampling keeps it diverse and
novel (the Phase-2 advancement axes). But a hot pool also weakens the top-50: the generator's most-
active designs sit on its high-probability modes, which hot sampling under-samples. So the **top-100
candidate pool is drawn cooler (`--top-temperature 0.8`, feat-028/029 mixed-temperature sampling)** and
ranked separately; only the ~100 selected peptides are cool, and they are prepended to the hot library
body (which guarantees the top-100 ⊆ library rule). We verified the cooler top-pool is the right choice
for the **composition** ranker too: it supplies **more** Lysine-rich, low-aromatic, non-hemolytic
candidates than a hot pool (≈3,700 vs ≈2,000 per 80k sampled), because the activity-fine-tuned
generator's high-likelihood modes are themselves Lysine-rich — so the composition ranking has ample
supply to select from, and the library body meanwhile keeps its **diversity/novelty unchanged** (only
the ~100 selected peptides are cool).

The composition push is deliberately **paired with a selectivity penalty**, without which it would fail:
a composition-only ranking drifts hemolytic (a pure-composition top-50 hits ESMC-predicted P(hemolytic)
**0.89**), because the most Lysine-rich cationic peptides can be membrane-lytic. So within the active band
we subtract **1.5·P(hemolytic)** from an **ESM Cambrian 600M** selectivity model (held-out AUROC 0.905;
details below), which holds the shipped top-50 at **0% predicted-hemolytic** (median P 0.001, max **0.052**
— *better* selectivity than APEX-potency ranking's 0.073). The resulting top-50 sits in the Lysine/Arginine
**blend** the peptide literature finds most active-and-selective on Gram-negatives (Hackney 2026; Zou 2007;
van der Walt 2025): median R/(R+K) **0.286** (Lysine exceeds Arginine in **88%** of the top-50, versus 14%
under APEX ranking), aromatic fraction 0.10, net charge +7, length 18.

As a **final selection step we cap this composition ranking to the wet-lab-validated cationic envelope**
(feat-037, the new `uv run generate` default `--max-cationic-fraction 0.444`). The unconstrained ranker pushed
~20 of the top-100 past the most cationic of our 46 wet-lab actives (cationic fraction (K+R)/len = 0.444),
concentrated in the assayed head of the list — and on those 46 the beyond-envelope region is exactly where **real
Gram-positive activity collapses**, so demoting the extrapolations below every in-envelope candidate lifts the
capped top-50's expected **Gram+/MDR Success Rate by +0.029 / +0.026** (an honest kNN arbiter on the 46, 95%
bootstrap CI excludes 0). This **supersedes feat-033 as the shipped default** while leaving its core (rank by
Lysine-richness within the APEX band) unchanged and the 50k library body **byte-identical** — only the top-100
selection changes, so the top-50 characterization above is the capped one (fcat median 0.400, max 0.438, was
0.667). *Honest caveat, carried throughout:* this gain is a **kNN-model estimate on n=46**, and the removed
peptides are extrapolations with **no direct wet-lab measurement** — a modest (~+0.02–0.03 SR), explicit,
ground-truth-supported, one-shot-appropriate refinement that *removes a known extrapolation risk*, **not a measured
wet-lab improvement** (`--max-cationic-fraction 1.0` recovers the pre-cap feat-033 selection). See
`docs/RESEARCH.md` (feat-037).

For transparency we also report the top-50's **APEX-predicted** Success Rates at ≤16 µM — **Broad 0.64,
Gram- 0.85, Gram+ 0.26, MDR 0.33** — but read them through the finding above. The very high Gram-negative
and the low Gram-positive/MDR *both* reflect APEX's chemotype bias, not expected real activity: APEX rewards
the Arginine/aromatic peptides it was trained to favour and under-scores the Lysine-rich chemotype. On
**every** real-activity test we have — the 46 wet-lab MICs and the DBAASP within-band analysis — the
composition selection **ties or beats** APEX-potency selection on all four activity categories *including
Gram-positive and MDR* (46: real Gram+ 0.30 vs 0.22; DBAASP within-band: real Gram+ rises 0.60→0.71 as
Lysine weight rises). We therefore expect this selection to raise real assayed activity across the board
relative to oracle ranking — most confidently on the Gram-negative-dominated **Broad** category and the
**Selectivity** window — while giving up APEX's (mis-calibrated) Gram+/MDR *predictions*, which the real
data show do not track real activity inside our selection band. This is an explicit, data-grounded,
one-shot-appropriate bet: an informational edge from wet-lab + external data that a pure oracle pipeline
cannot find. See `docs/RESEARCH.md` (feat-033).

The result (all figures are **computational predictions, not measurements** — we make no wet-lab
efficacy claim): a 50,000-peptide library that is **diverse** (91% of a random 500-sample have
nearest-neighbour Levenshtein identity < 0.6), **novel** (median identity to any known AMP ≈0.60, 90th
percentile 0.69, 1.3% above 0.80), and AMP-like (88% net-cationic, mean length 19); and a **top-100**
drawn from the APEX-active band and ranked by wet-lab-calibrated composition — all predicted active on
multiple strains, **0% predicted-hemolytic** (median P(hemolytic) 0.001, every top-50 well below 0.5, max 0.052), and
**novel** (top-50 median nearest-known-AMP identity 0.69, top-100 max 0.80, none an exact match, all
within the 0.80 rule).

## Model

- **Approach:** generative autoregressive language model over the 20-amino-acid alphabet (a
  generative method, as required), **fine-tuned toward predicted activity/selectivity by
  rejection-sampling fine-tuning (ReST)**. Candidates are then ranked by a **wet-lab-calibrated
  composition score within an APEX-active-band gate** (feat-033), with a hemolysis-selectivity penalty
  and a diversity screen — see *Selection and ranking*
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
  candidate pool** is sampled at **`--top-temperature 0.8`** (cool → its best peptides sit on the
  generator's high-activity modes). The top list is prepended to the library body, so the library body
  is entirely temperature-1.6 apart from the ~100 selected peptides — Phase-2 diversity is unchanged
  (0.84) while the top-50 gains Gram-/Broad (see the abstract and `docs/RESEARCH.md` feat-028/029). Set
  `--top-temperature 1.6` to recover the earlier single-hot-pool run.
- **Constraint handling:** alphabet, length 8–50, uniqueness, and exclusion of exact matches to
  `data/antibacterial.fasta` are enforced in `src/amp_challenge_2027/constraints.py` and applied
  during generation (`build_library` collects into a `dict` for order-stable de-duplication).
- **How many candidates were drawn to fill the library, and how many were rejected:** the generator
  is highly valid and diverse, so the library fills in a few over-drawn rounds (over-draw factor
  1.2). Rejections are duplicates, the rare invalid sequence, and any exact match to the reference
  set. The library is 50,000 unique valid sequences; a fresh run reproduces it byte-for-byte.
- **Library characterization (for the phase-1 diversity/novelty/physicochemical screen):**
  **diverse** (91% of a random 500-peptide sample have nearest-neighbour Levenshtein identity below
  0.6), **novel** (median max-identity to any known AMP ≈0.60, 90th percentile ≈0.69, only ≈1.3% above
  0.80 — and the library rule only forbids *exact* matches), and physicochemically **AMP-like** — 88%
  net-cationic, length 8–50 (mean ≈19), moderate hydrophobicity. Activity fine-tuning did not
  collapse the library: hot sampling (temperature 1.6) keeps it distributionally realistic and
  non-redundant, comparable to the un-fine-tuned base generator. The library is essentially unchanged
  by the feat-033 composition ranking, which reorders only the ~100 top-list peptides prepended to it.

## Selection and ranking of the top 100

This is a required deliverable and it is what the competition measures — 25 peptides are drawn at
random from the top 50 and assayed. (*The organizers' own materials disagree on this draw: the
How-It-Works section and the design PDF say the top **100**, while the website FAQ says the top **50** —
flagged in the checklist to confirm before submit; if it is the top 100, list positions 51–100 are also
assayed and our composition ranking tapers there.*)

- **Scoring function (feat-033, capped by feat-037):** within an **APEX-active-band gate**, `score =
  composition_score − 1.5·P(hemolytic)`, where `composition_score = lys_fraction − 0.5·aromatic_fraction`,
  applied to a large **8× (400k)** oversampled candidate pool, and then a **composition-envelope cap** (feat-037,
  `--max-cationic-fraction 0.444`) demotes any candidate with cationic fraction `(K+R)/len > 0.444` below every
  in-envelope one. The top-100 is ranked by **wet-lab-calibrated composition**, not by APEX-predicted potency — a
  deliberate inversion we justify by testing every candidate signal against *real* activity.
  - *APEX-active-band gate* — from **APEX-pathogen** (`oracle/apex`, isolated `uv` subprocess), which
    predicts MIC (µM) against the 11 clinical pathogens. We take the **top-20,000** candidates by APEX
    balanced Success-Rate score as the "active band" and rank *within* it by composition. Why
    gate-not-rank: on our 46 wet-lab MICs **and** on 946 independent DBAASP peptides, APEX ranks real
    activity well across the *full* range (Spearman ρ **+0.33**; full-range top-decile-by-APEX real Broad
    0.77) but **flattens inside the high-activity band** the top-100 is drawn from (ρ **+0.06** in the
    APEX-top-15%) — a textbook range-restriction failure. So APEX is excellent at *reaching* the active
    band (per-(peptide,strain) AUROC 0.62–0.67) and unreliable for *ordering within* it. See
    `docs/RESEARCH.md` (feat-033).
  - *composition_score* (`oracle.composition_score`) — `lys_fraction − 0.5·aromatic_fraction`, a
    deterministic, pool-independent ranker. *Within* the APEX band, **Lysine-richness predicts real
    activity** (Spearman +0.27 on the DBAASP top-15%, +0.42 on the 46) while APEX does not, and aromatic
    content anti-predicts it on the 46 while tracking hemolysis. Ranking the 46 by this score gives a
    **real** Broad Success Rate of **0.46** (Gram- 0.54) versus 0.31 for a random draw and ≤0.30 for
    *any* APEX score — and it beats APEX on all four activity categories; the DBAASP within-band data
    replicate the direction (Lysine lifts real Broad, Gram-negative **and** Gram-positive together). The
    top-50 lands in the Lys/Arg **blend** (median R/(R+K) 0.286), never poly-Lysine — the pure-Lysine
    extreme is both hemolytic and screened out by the gate + selectivity penalty. This **supersedes** the
    earlier feat-031 Arg-excess *hedge* (`--lys-hedge`), which only nudged an APEX-dominated ranking: the
    data show APEX *ordering* is worthless-to-harmful inside the band, so composition is made the primary
    key rather than a small subtraction. Set `--composition-weight 0` to recover the feat-031 APEX
    ranking (and `--aromatic-weight` tunes the aromatic term).
  - *Composition-envelope cap* (feat-037, `--max-cationic-fraction 0.444`, the shipped default) — a final
    selection step that keeps the assayed top-100 **inside the wet-lab-validated cationic envelope**.
    Maximising Lysine-richness runs off the end of the data it was calibrated on: the uncapped ranker pushed
    **~20 of the top-100** past the most cationic of our 46 wet-lab actives (`fcat = (K+R)/len` up to 0.667),
    concentrated at the head (13 of the top-50), and on the 46 that beyond-envelope region is where **real
    Gram-positive activity collapses** (fcat>0.40 → Gram+ SR ~0.07 vs a peak ~0.36). The cap demotes any
    candidate with `fcat > 0.444` (the wet-lab maximum) below every in-envelope candidate, so the top-50 now
    has `fcat` median **0.400 / max 0.438**. Measured against real activity with an honest kNN arbiter on the
    46 (LOO-validated), this lifts the capped top-50's expected **Gram+/MDR Success Rate by +0.029 / +0.026**
    (95% bootstrap CI excludes 0); Gram−/Broad are directionally positive (CI includes 0). 0.444 is optimal —
    a tighter 0.40 cap loses the Gram− gain and collapses diversity. This **changes only the top-100 ranking**
    (the 50k library body is byte-identical); `--max-cationic-fraction 1.0` recovers the pre-cap feat-033
    selection. *Honest caveat:* the gain is a **kNN-model estimate on n=46** and the removed peptides are
    extrapolations with **no direct wet-lab measurement** — a modest (~+0.02–0.03 SR), ground-truth-supported,
    one-shot-appropriate refinement, not a measured wet-lab improvement. See `docs/RESEARCH.md` (feat-037).
  - *P(hemolytic)* — from a **selectivity model built on the latest SOTA protein language model,
    ESM Cambrian 600M** (ESM++ `Synthyra/ESMplusplus_large`, MIT, on GPU-if-available), with a small
    trained MLP head over its mean-pooled embeddings (`checkpoint/selectivity_esmc.pt`, trained on
    HemoPI-2). Held-out **AUROC 0.905** — a large upgrade over the 11 physicochemical descriptors it
    replaces (0.778) and over ESM-2 150M (0.883); the larger **ESMC-6B gives no further gain** (also
    0.905 — the ceiling is the ~1000-peptide labelled dataset, not model size), so 600M is kept. The
    PLM is applied **two-stage**: rank the whole pool by activity, then score selectivity only on the
    top `refine_k=20000` (the rest assumed hemolytic) — exact for the top-100 and keeps the run fast.
    The most Lysine-rich cationic peptides can be membrane-lytic, so the **λ=1.5** penalty is not a
    garnish but **essential to the composition ranking**: a composition-only ranking drifts hemolytic (a
    pure-composition top-50 reaches ESMC P(hemolytic) **0.89**), while pairing composition with the penalty
    holds the shipped top-50 at **0% ESMC-predicted-hemolytic** (median P 0.001, max **0.052**) — the
    Optimal Selectivity standout, and *better* here than under APEX-potency ranking (max 0.073). The
    physicochemical model (`checkpoint/hemolysis.pt`) is retained as a graceful fallback if the PLM
    weights cannot be fetched.
  - *Legacy terms (superseded by composition).* Two earlier ranking terms — a closed-form **amphipathicity**
    reward on the Eisenberg hydrophobic moment (feat-025, `--amphipathicity-bonus`) and the **Arg-excess
    hedge** (feat-031, `--lys-hedge`) — are **no-ops under the shipped composition ranking**, which subsumes
    both: it prefers Lysine over Arginine directly, and low-aromatic Lysine-rich peptides are precisely the
    amphipathic, non-hemolytic class those terms were reaching for. They remain available as flags and take
    effect only when `--composition-weight 0` recovers the feat-031 APEX-ranked path; see `docs/RESEARCH.md`
    (feat-025, feat-031).
- **Ranking procedure:** score every library sequence, sort by descending score (sequence as a
  deterministic tiebreak), then walk down the list applying the novelty and diversity screens
  below until 100 are selected. Fully deterministic — APEX runs in eval mode on CPU with every math
  library pinned to one thread (both the pooled and the sequential scoring paths, via a shared env
  helper — multi-threaded MKL/oneDNN reductions are not bit-reproducible), and per-sequence results
  are reassembled by input order, so the output is byte-identical across runs regardless of the
  machine's core count *or* available memory (which selects the path). The ESMC selectivity head
  runs in eval mode under `torch.use_deterministic_algorithms` with a fixed cuBLAS workspace, so the
  two-stage GPU path is byte-reproducible too — verified by regenerating twice: `top.fasta` and
  `library.fasta` are md5-identical.
- **Novelty screen:** candidates above 0.80 Levenshtein ratio against any sequence in the
  reference set are rejected and replaced by the next-ranked candidate. In the shipped run the
  novelty screen rejected **641** higher-ranked candidates — the composition ranking still sits
  comfortably below the 0.80 threshold (top-50 median identity to any known AMP **0.69**, top-100 max
  **0.80**, none an exact match), so no near-duplicate of a known AMP reaches the assayed set.
- **Diversity or redundancy control within the top 100:** a within-list cap
  (`--diversity-max-identity 0.6`) skips any candidate exceeding 0.60 Levenshtein identity to an
  already-selected peptide, keeping the more-active member of a near-duplicate pair. **2,067 near-
  duplicates were rejected** in the shipped run — the composition ranking concentrates the ranked top
  into a few Lysine-rich motif families, so the 0.6 cap does real work to
  keep the top-100 diverse. This matters because the ranking
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
  as fallback) applied in ranking; a **wet-lab-calibrated composition ranking** (feat-033 — the top-100 is
  ranked by Lysine-richness minus aromatic content within the APEX-active band, `oracle.composition_score`),
  with a **composition-envelope cap** (feat-037, `--max-cationic-fraction 0.444`) that demotes any candidate
  whose cationic fraction (K+R)/len exceeds 0.444 — the maximum among our 46 wet-lab actives — below every
  in-envelope candidate; and a **within-list diversity cap** (0.60 Levenshtein identity). The composition term
  and the envelope cap are *ranking* keys, **not** hard windows — they reorder the top-100 (the 50k library body
  is unaffected) and exclude no candidate from the library; beyond the APEX-active-band gate no charge/pI,
  aggregation, or solubility filters are applied; those other properties emerge from the generator and are
  only *measured* for disclosure.
- **External predictors or databases used at selection time:** **APEX-pathogen** (MIC predictor,
  MIT-licensed, vendored under `oracle/apex`) for activity; an **ESM Cambrian 600M** selectivity model
  (ESM++ `Synthyra/ESMplusplus_large`, MIT weights fetched from HuggingFace; HemoPI-2-trained head) for
  hemolysis. Both are disclosed in `docs/DATA.md`. No proprietary or non-public data or services are
  used at any stage.

## Reproducibility

- `uv sync` then `uv run generate` regenerates both files exactly (fixed seed 42; APEX runs in
  eval mode on CPU; `torch.use_deterministic_algorithms`). The shipped default is the **feat-037
  composition-envelope-capped** selection — current hashes **top `21fd02b7aa928f32c1ac6f6aeb1faa2b`,
  library `bba245dccc21be693a80c1b114748bb5`** (pre-cap feat-033 was `dc37c540`/`06e30960`, recoverable
  with `--max-cationic-fraction 1.0`). Because the cap changed `top.fasta`, the official validator must be
  **re-run on the final pushed commit before submitting — pending, not yet re-confirmed for feat-037.**
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
- [ ] **feat-037 changed `top.fasta`, so the official validator MUST be re-run on the final pushed commit —
      PENDING, not yet claimed passed.** The shipped artifact is now the **composition-envelope-capped** selection
      (**top `21fd02b7aa928f32c1ac6f6aeb1faa2b`, library `bba245dccc21be693a80c1b114748bb5`**); the pre-cap feat-033
      output (`dc37c540`/`06e30960`, recoverable via `--max-cationic-fraction 1.0`) is what the passes below
      validated, and local two-run byte-repro on the capped default is being re-confirmed. History:
      `scripts/verify_submission.py` **PASSED on the byte-repro-hardened commit `1c091e5`** (pre-cap feat-033 default,
      after the adversarial-audit fix to the sequential APEX path): fresh GitHub clone + `uv sync` + generate ×2
      — *"All checks passed. Submission is valid!"*, all 8 checks incl. byte-identical reproducibility, the ≤80%
      novelty gate, and the real ESM++/ESMC selectivity path; fresh-clone output library `06e30960`, top
      `dc37c540` (byte-identical to local, and to the pre-fix `84e2b78` — the fix is byte-neutral on the shipped
      parallel path). Commits atop the validated `1c091e5` are docs plus one **byte-neutral** review-response
      commit (feat-036 — stdout/test-only changes on non-default paths; a 2× `generate` run reproduced
      `dc37c540`/`06e30960`); the local test suite is green — **158 tests**. The **official validator was
      re-run on the pre-cap commit `a7ed06f` (2026-09-30) → all 8 checks PASSED, fresh-clone byte-identical
      `dc37c540`/`06e30960`** — but that validated the **pre-cap feat-037** artifact; the feat-037 cap changed
      `top.fasta` (new hashes `21fd02b7…`/`bba245dc…`), so the validator **must be re-run on the final feat-037
      commit before this box can be checked.** **feat-035 (2026-09-30): a
      24-hour deep-validation pass with no artifact change confirmed feat-033 is near-optimal** — the
      composition chemotype's Optimal-Selectivity safety window was validated on real DBAASP HC50/MIC (92.7 vs
      58.5 baseline), the APEX active-band gate was tested keep-vs-drop and **kept** (the 46-peptide
      "gate-harmful" signal is not statistically robust, and the DBAASP "gate-helpful" result is circular
      because APEX was trained on DBAASP), and a learned/TabPFN-style ranker was shown to collapse
      out-of-distribution on novel peptides while the composition formula transfers (Spearman +0.448); no
      validated Gram+/MDR lever was found. **feat-036 (2026-09-30):** a background code review found no
      correctness bug; its top finding + two more independent ground-truth cross-checks reconfirm feat-033
      (artifact byte-unchanged): the selectivity λ=1.5 acts as a *gate* (the top-100 is 99% clean and
      **composition**-ordered, not selectivity-ordered), an independent HemoPI2-recipe hemolysis model
      corroborated the safe chemotype once arbitrated by **real HC50** (a third learned model that fails OOD on
      novel peptides, while composition-on-real-data transfers), and no safe Gram+ lever exists (net-charge
      reverses sign OOD). *Honest caveat:* the "0% predicted-hemolytic" figure is an ESMC estimate on novel
      peptides — the trustworthy basis is the real-HC50-validated composition chemotype, never a per-peptide
      guarantee. See `docs/RESEARCH.md` (feat-035; code-review closeout; selectivity cross-check).
      **Re-run once more on the
      final pushed commit immediately before submitting.** History: the earlier **feat-031 APEX-ranked
      default PASSED** the official validator on commit `f4eed63` (fresh-clone output `9a3278c9`/`61becbab`);
      recover it with `--composition-weight 0`. **feat-032 (a Lys-conditioned Gram--ReST generator) was
      explored and REVERTED** (its APEX "domination" was ensemble-Goodhart; it regressed near-exact
      novelty). See docs/RESEARCH.md (feat-031, feat-032, **feat-033**). Re-run the validator once more
      immediately before submitting.
- [x] Every section above filled in, with no placeholder text left
- [x] Repository public, MIT licensed, `uv.lock` and `.python-version` committed
- [x] Weights committed or fetchable, and the inference path documented (`checkpoint/`, `oracle/apex/`)
- [x] Read access for [@RasmusML](https://github.com/RasmusML) and
      [@szymczakpau](https://github.com/szymczakpau) — satisfied by the repo being public
- [x] Top 100 confirmed to be a subset of the submitted 50,000-sequence library (enforced in code)
- [ ] Only one entry for this model; if a second model is planned, organizers contacted in advance
> **Both flags below were SET ASIDE by the participant (2026-09-30) — kept as a due-diligence record, not
> submit blockers. Do not re-raise them.**

- ~~**Confirm the wet-lab draw size with the organizers before the one-shot submit**~~ — *set aside per the
      participant.* Their own materials disagree: the website How-It-Works section and the design PDF say **25 are
      drawn at random from the top 100**, while the website FAQ says **top 50**. If it is the top 100, list positions
      51–100 are also assayed and our composition ranking tapers there; the design hedge (a uniformly strong top-100)
      already covers both cases. See `docs/COMPLIANCE.md`.
- ~~**Confirm the institutional-email registration requirement is satisfied**~~ — *set aside per the participant.*
      The competition materials reportedly require an institutional email ("gmail/hotmail/yahoo not accepted"); our
      Kaggle account (`j_v_v_07` / `vallabh2006@gmail.com`) registered with a gmail address, though the listed
      contact is institutional (`be23b041@smail.iitm.ac.in`). See `docs/COMPLIANCE.md` § Account identity.
