# SOTA research notes (2026-09-27)

Research pass before committing the model architecture, per the "use latest after researching"
directive. Sources are the competition's own starter kits and the recent literature.

## The organizers' SOTA baseline: AMP-Diffusion + APEX

The official `szczurek-lab/ampdiffusion-starter-kit` is the reference SOTA pipeline (an *excluded
baseline* — organizers run it, so it can't win, but its method is the bar to beat):

- **Generator:** AMP-Diffusion (Torres et al., *Cell Biomaterials* 2025; Chen et al., bioRxiv 2023;
  code `programmablebio/amp-diffusion`) — latent diffusion in **ESM-2 (esm2_t6_8M)** embedding
  space; a `Denoise_Transformer` + `GaussianDiffusion1D`, 16.54M-param EMA checkpoint, 1000-step
  sampler, decoded to sequences via the ESM-2 LM head. Deps: `fair-esm`, `torch==2.5.1`, `einops`,
  `ema-pytorch`. **Requires a CUDA GPU** — "a full CPU run of the sampler is impractically slow."
- **Ranking (top-100), the paper's filtering — no optimization/curation:**
  1. **APEX** ensemble (8 models) predicts mean MIC across an **11-pathogen panel**; take
     most-potent-first, preferring **MIC ≤ 64 µM**. APEX = `machine-biology-group-public/apex-pathogen`
     (de la Fuente lab, *Nat. Microbiol.* 2025) — the **same lab that runs this competition's wet
     lab**, so it is the best available proxy for the actual MIC measurements.
  2. **Novelty:** drop > 0.60 Smith-Waterman local-alignment similarity to known AMPs.
  3. **Diversity:** for any pair > 0.40 SW-similar, keep the lower-MIC one.
  4. Final guard: < 80% Levenshtein vs `data/antibacterial.fasta`.
- **Bundled data:** `data/training/training.fasta` (DRAMP 3.0 + APD3 + DBAASP) and
  `experimental/mic.csv` — labeled MIC data (addresses feat-007).

The `hydramp-starter-kit` is the older HydrAMP VAE (TensorFlow 2.2) — not the frontier.

## Wider literature (2024-2025)

- AMP-Diffusion (latent diffusion over ESM-2) is the current go-to for AMP generation.
- The de la Fuente lab's 2025 latent-diffusion LM (Science Advances, `sciadv.adp7171`) reports
  training-set similarity as low as ~0.57 and 25/40 synthesized peptides active — novelty +
  potency together, which is exactly what this competition scores.
- Protein-LM generators in play: ProtGPT2, ProGen (autoregressive), EvoDiff (diffusion). ESM-2 is
  the standard embedding/decoder backbone; newer ESM-C/ESM-3 exist but the whole APEX + AMP-
  Diffusion ecosystem here is ESM-2-based, so ESM-2 is the pragmatic backbone for compatibility.

## Decisions for this repo

1. **Shipped generator = our own CPU-runnable model** (feat-006/011). Rationale: AMP-Diffusion's
   sampler needs a GPU, but the validator may be CPU-only; our small AR-Transformer samples on CPU
   deterministically. AMP-Diffusion is used **offline** as an additional candidate source (feat-012).
2. **Ranking oracle = APEX** (feat-013), not a from-scratch predictor — organizer-provided and
   wet-lab-aligned. Integrated in its own isolated `uv`/subprocess env, as the starter kit does.
3. **The edge over the baseline = optimization** (feat-014): the baseline only *filters* by APEX;
   we additionally *optimize* candidates for low APEX-MIC and low hemolysis (selectivity), plus
   diversity — using the 8 H100s for large-scale generation + scoring + directed refinement.
4. **Labeled data (feat-007):** reuse the starter kit's `training.fasta` + `mic.csv` (public:
   DRAMP 3.0 / APD3 / DBAASP), with provenance in `docs/DATA.md`.

Everything above stays subject to the honesty guardrails in `docs/MODEL_PLAN.md`: predicted MIC is
APEX's estimate, not a measurement, and we report novelty/diversity and APEX validation metrics.

---

## APEX validation (2026-09-27, feat-007/013) — measured, not assumed

Before ranking anything by APEX, I checked it against the **46 wet-lab-measured peptides** in
`data/experimental/mic.csv` (Torres et al. 2025) and against broad discrimination. Reproducible
via `oracle.ApexScorer`. Findings drive the selection design:

| Test | Result | Reading |
|---|---|---|
| Per-(peptide,strain) inhibition AUROC (n=506, novel peptides) | **0.62** | moderate on novel sequences |
| Known-AMP vs length-matched random, AUROC (n=3000) | **0.76** | good at reaching the active band |
| Per-peptide mean-MIC Spearman on the 46 | ~0.0 | **range restriction** — the 46 are all APEX-selected actives (mean-MIC 44–63 µM vs the AMP population's 146–506), so correlation collapses there; not evidence APEX is useless |
| Canonical AMPs | LL-37 min 6.4 / poly-G min 431 | ranks actives above inactives; **min-MIC (best pathogen) discriminates; mean-MIC over-penalises** real actives (magainin, indolicidin) |
| Absolute scale | median mean-MIC ~430 µM for broad corpus | trust **relative rank**, not absolute µM |

**Likelihood vs APEX on a real 20k library from our shipped generator:** the likelihood-ranked
top-100 sits at only the **29th percentile** of APEX potency (median min-MIC 178 µM, 10% ≤32 µM)
and shares **0/100** with the APEX-ranked top-100 (median min-MIC **2.6 µM**, 100% ≤32 µM). The
generator already emits ~10% of candidates at min-MIC ≤32 µM, so activity is *selectable*, not
absent. Switching the ranking from likelihood to APEX is the single biggest lever.

**Design consequences (guarding against Goodhart on a moderate oracle):**
1. Rank by APEX but **do not select the global APEX-minimisers** — that is the unreliable tail.
   Select a **diverse, novel, low-hemolysis** set from the active band.
2. Aggregate by **min / low-quantile MIC** (+ a breadth count), not naive mean.
3. **Selectivity (hemolysis) is orthogonal** to APEX and additive — the baseline ignores it.
4. Cross-check with physicochemistry; report the novelty-rejection and diversity stats honestly.

## Shipping architecture decision (feat-014)

The organizers' own baseline runs APEX **live inside `generate`** as a vendored isolated `uv`
subprocess (CPU scoring, deterministic), with weights committed. That is the sanctioned pattern,
so we adopt it: **library generated live** (fast, diverse, novel); **top-100 ranked live** by
APEX + hemolysis + Smith-Waterman novelty(>0.60)/diversity(>0.40), matching the paper's recipe
and adding the selectivity axis. The 8×H100s are used **offline** to improve the shipped
generator/selection policy (ensemble, conditioning, threshold tuning), not to bake a static
output. The current likelihood ranking stays the default until the APEX path passes the full
gate + the official validator on a fresh clone.
