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

## Selectivity / hemolysis (2026-09-27, feat-013)

Optimal Selectivity is scored by the safety window HC50/MIC50, and APEX favours hyper-cationic,
hydrophobic peptides — the hemolysis-prone kind — so an orthogonal hemolysis signal both opens
that category and de-risks the top-100. A lean 11-descriptor MLP (`physchem` features, ships in
the main env, deterministic CPU) predicts P(hemolytic).

**Dataset bias, caught and fixed.** Trained on HemoPI-1 (hemolytic vs *random* non-hemolytic) the
model hit held-out AUROC 0.987 but scored 99% of our APEX-active peptides ~1.0 — it had learned
"AMP-like ⇒ hemolytic" (the negatives are non-AMP random fragments), useless for ranking among
actives. Retrained on **HemoPI-2** (high vs low hemolytic potency, both real peptides): held-out
AUROC 0.778, and a genuine spread among active peptides (P percentiles 0.13/0.41/0.75/0.95/0.99).
Moderate accuracy → used as a *soft* signal, not an authority.

**Activity/selectivity trade-off (tuned on a 20k pool).** Ranking by
`broad_potency − λ·P(hemolytic)`: at λ=2 the top-100 stays 100% active with mean predicted
breadth 5.26→4.62 (~12% cost) while mean P(hemolytic) halves 0.73→0.36 and strongly-selective
peptides (P<0.3) rise 12→53. λ=2 is the shipped default (4 of 5 categories are activity and the
hemolysis signal is noisier than APEX, so the nudge is deliberately modest); tunable via
`--hemolysis-penalty`.

**Upgrade to a PLM selectivity model (2026-09-28, feat-017).** Following the directive to use the
latest and largest models, we replaced the 11 physicochemical descriptors with embeddings from the
**latest SOTA protein language model, ESM Cambrian 600M** (ESM++ `Synthyra/ESMplusplus_large`, MIT,
transformers-native so it needs no torchtext) plus a small trained MLP head on HemoPI-2. Held-out
**AUROC climbs 0.778 → 0.905** (an intermediate ESM-2 150M gave 0.883), so the signal is now strong,
not merely soft. A devil's-advocate audit with this accurate model exposed a real problem in the
previous physicochemical-selected top-100: **~51% of it was in fact predicted hemolytic** (median
P 0.54) — the 11-descriptor model had been over-optimistic about its own selections. Re-ranking with
the PLM model (two-stage: score selectivity only on the top `refine_k=4000` most-active candidates, at
λ=0.5) drives the top-100 to **0% predicted hemolytic** (median P 0.006) **while breadth slightly
rises** (category-success 0.884 → 0.914, MDR Success-Rate 0.36 → 0.38) — a Pareto improvement, not a
trade-off. Inference is byte-deterministic on GPU (`use_deterministic_algorithms` + fixed cuBLAS
workspace; regenerating twice gives md5-identical `top.fasta`/`library.fasta`), and if the weights
cannot be fetched `generate` falls back to the physicochemical model. `training/train_selectivity_esmc.py`
trains the head; `src/amp_challenge_2027/selectivity_esm.py` serves it. Shipped as the default.

---

## Activity/selectivity fine-tuning by rejection sampling (ReST) — session 3 (2026-09-28)

**Problem.** The pre-trained generator wastes ~94% of its samples: only ~6% are predicted active
(APEX MIC ≤16 µM on ≥1 strain). Ranking finds the rare actives, but the *distribution* is weak.

**Method.** Rejection-sampling fine-tuning on the 8 H100s. Each round: (1) sample ~200k peptides;
(2) score all with APEX (sharded across the 8 GPUs via a CUDA build of APEX's torch, offline only —
the shipped path stays CPU-deterministic) and the hemolysis model; (3) keep the highest-reward
*novel, active* peptides; (4) continue training at low LR with a corpus anchor. Reward = the
category success-rate score (Gram+/MDR up-weighted) − λ·P(hemolytic).

**Result (predicted, not measured).** Whole-pool active fraction 6% → ~75% over five rounds; top-50
mean predicted breadth ~3.9 → ~6 of 11 strains, Gram-negative Success Rate to ~70%, P(hemolytic)
0.29 → <0.1. A parallel sweep over λ and the Gram+/MDR up-weight mapped the tradeoff: the up-weight
rebalances Gram-negative↔Gram-positive; λ trades breadth for selectivity.

**Anti-Goodhart checks (all pass).** (a) The learned peptides are realistic cationic amphipathic
α-helices (e.g. `ILGKLLSTAAKLLSKL`, charge +3..+4, length 15–18), not adversarial noise. (b) On
APEX's eight *independent* sub-models, ~95% of "active" calls are backed by ≥6/8 sub-models, and
breadth via sub-models {0-3} vs {4-7} agrees (corr 0.67) — the activity is not an artifact of gaming
the ensemble mean. (c) The physicochemical envelope stays in the real-AMP range. (d) Verbatim
novelty is preserved. APEX remains a *moderate* signal, so selection also layers novelty, diversity
and a hemolysis penalty; we make no wet-lab efficacy claim.

**Diversity/novelty vs activity — the temperature fix.** Fine-tuning concentrates the sampling
distribution: over rounds, within-pool diversity collapses (a top-region diversity proxy fell
150 → ~17) and median identity to known AMPs rose (0.51 → ~0.72), which would hurt the phase-1
diversity/novelty screen. Capping near-duplicates in the fine-tune set barely helped. What worked is
**sampling at an elevated temperature**: on a mid ReST round, temperature 1.6 restored library
diversity (≈30% → ≈79% of a random sample mutually <0.6 identity) and novelty (median identity to
known 0.73 → 0.50) **with no measured top-50 activity cost**, because the top-100 is still chosen by
APEX from a now-broader active manifold. The shipped generator is therefore a mid ReST round sampled
at temperature 1.6.

**Shipped ranking.** Switched from the unbounded `broad_potency` margin to `category_success_score`
(mean saturating soft-Success-Rate, Gram+/MDR up-weighted) − 0.5·P(hemolytic): threshold-focused
(rewards clearing 16 µM on many strains, not sub-µM depth on a few), and balanced across the five
scored categories. CPU-APEX is now sharded over single-threaded workers — byte-reproducible and
independent of core count (feat-016), which makes a larger oversample affordable at ship time.

### Independent cross-check with Macrel (2026-09-28)

To avoid betting solely on APEX + our own hemolysis model, we cross-checked the shipped top-100
against **Macrel** (Santos-Júnior et al.) — an independent AMP/hemolysis classifier trained on
different data. **Activity is independently corroborated:** Macrel calls 100% of the top-100 an AMP
(median P(AMP) 0.71), consistent with the APEX-based selection. **Hemolysis: the two predictors
disagree, and calibration shows Macrel's is the unreliable one.** On 1,014 *labeled* HemoPI-2
peptides, Macrel calls 99% of known-hemolytic *and* 93% of known-**non**-hemolytic peptides hemolytic
(≈no specificity), whereas our HemoPI-2 model separates them (87% sensitivity, 78% specificity,
consistent with its 0.778 AUROC). So Macrel's "all hemolytic" verdict is uninformative and is *not*
ensembled into selection; our discriminating model is retained. Takeaway: the activity claim is
robust to an independent tool; the selectivity claim rests on a moderate but genuinely discriminating
model and is reported as such — no wet-lab claim is made.

### Structural cross-check with ESMFold2 (2026-09-28, feat-018)

A third, structure-based cross-check, orthogonal to the two sequence oracles. Active AMPs act by
folding into amphipathic α-helices that partition into bacterial membranes, so a peptide the oracles
call active *should* adopt a confident helix. We folded the shipped top-100 with **ESMFold2-Fast**
(`biohub/ESMFold2-Fast`, ESMC-6B backbone, 6.5B params, MIT — the latest SOTA single-sequence
structure predictor, honoring the directive to use the largest available model) and measured, per
peptide, mean pLDDT, α-helix fraction (from backbone φ/ψ of the predicted all-atom structure), and
the Eisenberg hydrophobic moment (amphipathicity). This is an **offline** check — ESMFold2 is a
diffusion model and non-deterministic, so it never enters the byte-reproducible selection path;
`experiments/structure_validate.py` reproduces it with a fixed seed.

**Result — the top-100 are confident amphipathic helices, with zero misfold flags.** Median mean-pLDDT
0.679, α-helix fraction median 1.00 (mean 0.98; every peptide >0.5), hydrophobic moment median 0.38 —
squarely in the range of canonical helical AMPs we folded as positive controls (melittin pLDDT 0.74 /
helix 0.96 / µH 0.35; magainin-2 0.75 / 1.00 / 0.45; LL-37 fragment 0.76 / 1.00 / 0.88), and far from
non-structured negatives (poly-G / poly-GS: helix 0.06, µH 0.00). **0 of 100** were flagged as
low-confidence-and-low-helix — no disordered, oracle-gaming artifacts.

**Honest caveat (devil's advocate).** Short peptides over-converge to helix in all current folders,
so a length-matched *random*-peptide control was also fairly helical (median helix 0.86). Helix
fraction is therefore **confirmatory, not discriminative**, at these lengths. The discriminating axis
is confidence: the top-100's median pLDDT (0.679) sits clearly above the random control (0.593) and
approaches the known-AMP band (0.74+). So the structural evidence confirms the designs adopt the
membrane-active amphipathic-helix fold characteristic of real AMPs and fold more confidently than
noise — it is corroboration, not a wet-lab claim.

## Balanced objective + large oversample: lifting the weak Gram+/MDR categories (2026-09-28, feat-019)

The competition scores five *separate* categories by the mean per-peptide Success Rate over the
assayed batch (25 drawn from the top 50). After the ESMC upgrade our top-50 was strong on
Broad/Gram- and Selectivity but weak on the two hard categories: **Gram+ SR 0.37, MDR 0.42**. Two
compounding causes, both fixed:

1. **The ranking objective diluted the hard-hitters.** `category_success_score` averages *soft*
   success across all 11 strains, dominated by the many (easy, for cationic peptides) Gram-negative
   strains; re-weighting Gram+/MDR within it plateaued at ~0.39 Gram+ *for any weights*. Ranking by
   **hard Gram+/MDR Success Rate** (`balanced_success_score` = SR_hard(GP) + SR_hard(MDR) + 0.5·mean
   soft) surfaces the peptides that actually *clear* the hard strains at <=16 uM.
2. **The pool was diversity-screen-starved of non-hemolytic Gram+/MDR actives.** At the old 3× (150k)
   oversample only ~19 diverse (<=0.60 identity) non-hemolytic Gram+/MDR-active peptides exist; this
   count grows ~linearly with pool size, so we raised the oversample to **8x (400k)** — affordable
   because feat-016's parallel CPU-APEX scores 400k in ~7.5 min.

**The Gram+/hemolysis tension, and the lambda fix.** Gram+-active cationic peptides are
overwhelmingly hemolytic (median ESMC P(hemolytic) **0.975** for Gram+ SR>=0.5 vs 0.028 otherwise) --
only ~30% are non-hemolytic. The `balanced` score's larger scale made the old lambda=0.5 hemolysis
penalty too weak, letting hemolytic Gram+ hitters into the top-50 (fraction with P>0.5 rose to 24%).
Raising the penalty to **lambda=1.5** restores **0% predicted-hemolytic** while keeping the gains --
we deliberately protect the Selectivity standout rather than chase the aggressive lambda=0.5 profile
(Gram+ 0.56 but 28% hemolytic).

**Result (top-50, the assayed set), vs the previous ESMC-selected submission -- a clean Pareto gain
on all five categories:** Broad 0.50->0.57, Gram- 0.57->0.60, **Gram+ 0.37->0.50, MDR 0.42->0.49**,
Selectivity held at **0% predicted-hemolytic** (median P 0.007->0.004). The new top-100 stays novel
(median identity to known 0.72, max 0.80, 0 exact matches) and -- cross-checked with ESMFold2 --
still folds into confident amphipathic helices (median pLDDT 0.68, helix 1.00, muH 0.40, 0/100
misfold flags), so the harder-optimised selection is not oracle-gaming. Byte-reproducible (two runs
md5-identical).

**Largest-model check (ESMC-6B).** Per the directive to use the largest models, we tested whether
ESMC-6B (`Synthyra/ESMplusplus_6B`, the largest ESM-C, 6.4B params) beats ESMC-600M on the HemoPI-2
selectivity task. It does **not** -- held-out AUROC 0.905 across three head seeds, identical to the
600M model (the ceiling is set by the ~1000-peptide labelled dataset, not embedding size). So we keep
ESMC-600M and avoid a ~10x inference cost for zero accuracy gain; the lever to push selectivity beyond
0.905 is more hemolysis *data*, not a bigger model.

## GP/MDR-targeted ReST with a selectivity gate: raising the hard-category ceiling (2026-09-28, feat-020)

feat-019 lifted Gram+/MDR by better *selection*, but the ceiling was the generator's own output --
only ~28 diverse non-hemolytic Gram+/MDR actives per 400k pool. To raise that ceiling we fine-tuned
the generator toward the rare target by rejection sampling (ReST) on the 8 H100s: sample -> score with
APEX (across all 8 GPUs) and the ESMC selectivity model -> keep the highest-reward peptides -> low-LR
fine-tune -> repeat.

**Reward design mattered -- the first attempt failed instructively.** A soft `activity - lambda*P(hemolytic)`
reward made Gram+/MDR *decline* over rounds (top-100 Gram+ 0.31->0.28): the generator raised reward the
easy way, by lowering predicted hemolysis, not by gaining the (biologically rare) Gram+/MDR hard-hits.
The fix was a **hard selectivity gate**: distil *only* from peptides the ESMC model calls non-hemolytic
(P<0.5) and reward *pure* Gram+/MDR hard Success Rate -- so the generator cannot trade activity for
selectivity, and the only way to raise reward is to actually gain Gram+/MDR activity. With the gate the
generator's whole distribution shifted toward activity over 4 rounds (20k-sample pool Gram+ Success Rate
7.6%->9.5%->..., MDR 10.1%->12.5%, active fraction 32%->40%) while predicted hemolysis stayed flat (~0.27)
and diversity held (div150=150, no mode collapse).

**Result -- a large further Pareto gain (top-50, the assayed set), over feat-019:** Broad 0.57->0.62,
**Gram+ 0.50->0.75, MDR 0.49->0.67**, Selectivity still **0% predicted-hemolytic** (median P 0.004->0.001),
at a small Gram- cost (0.60->0.54 -- a *selection* effect: the balanced objective picks Gram+/MDR
specialists, though the generator's Gram- output actually rose). Crucially the designs are **more novel**
(median identity to known AMPs 0.62 vs 0.72, max 0.73 vs 0.80) -- the harder optimisation is *discovering*
new Gram+/MDR motifs, not memorising known ones.

**Anti-Goodhart -- all four checks pass.** (a) Novelty *rose*, not fell. (b) 0% hemolytic per the
*independent* ESMC model. (c) The Gram+/MDR activity is real per APEX's own internal consensus: on the
top-50, **86-89% of APEX's 8 ensemble members agree** on the active Gram+/MDR calls (82-86% with >=6/8
agreement; median per-call log-MIC CV 0.37) -- not ensemble-mean-gaming. (d) ESMFold2 folds all 100 into
confident helices (median pLDDT **0.702**, *higher* than feat-019's 0.683; helix 1.00; 0/100 misfold
flags). One honest caveat: the Eisenberg hydrophobic moment is lower (0.29 vs 0.40) -- the ReST found a
somewhat less classically-amphipathic Gram+/MDR motif class; it is still within the real-AMP range
(melittin 0.35) and every hard check passes, but we flag it. As always these are *predictions* (APEX is
only moderate on novel peptides, AUROC 0.62), so the real-world gain carries the usual oracle-transfer
uncertainty and we make no wet-lab claim. The shipped generator is gated-ReST round 3; feat-019's
generator is preserved (git history at `60afc5f`, and `experiments/cache/`).

## Recovering the weakest category: an explicit hard Gram- term (2026-09-28, feat-021)

feat-020's generator + balanced selection is a Gram+/MDR *specialist* -- and the balanced objective
ranked only by hard Gram+/MDR Success Rate plus a broad soft tie-break, so the top-50 came out slightly
low on Gram- (0.54), the weakest of the five scored categories. Because the five categories are ranked
*separately* and a team's overall standing is the arithmetic mean across them, the **weakest category is
the highest-leverage place to improve** -- lifting a 0.54 floor helps the mean more than pushing an
already-strong 0.75 higher.

The fix is a one-line objective change: add a modest hard **Gram-negative** Success-Rate term
(`gn_weight=0.75`) to `balanced_success_score` (mirrored in `ApexRanker` and a `--gn-weight` flag). No
new model, no re-sampling -- the feat-020 400k pool **already contained** Gram-strong non-hemolytic
peptides; the objective simply needed to value them. A sweep (`experiments/gn_rebalance.py`) put the
knee at 0.75: enough to recover Gram- without displacing the Gram+/MDR specialists.

**Result (top-50, shipped CPU-APEX + ESMC) -- a strict, free improvement over feat-020:** Gram-
**0.543 -> 0.583** (+0.040), Broad **0.618 -> 0.644** (+0.026), Gram+ **0.750** and MDR **0.667** held
*exactly*, Selectivity still **0% predicted-hemolytic** (median P 0.001 -> 0.004, max 0.06 -- all far
below the 0.5 gate). The regenerate re-selected 45 of the 50 top peptides. **Rejected alternative:** an
*ensemble* of the feat-019 (Gram-strong) and feat-020 (Gram+/MDR) generator pools
(`experiments/ensemble_pool.py`) did **not** help -- balanced selection still picks feat-020's
specialists (only 5/50 of the top-50 came from feat-019) and Gram+ fell to 0.73; the single generator +
`gn_weight` is strictly better. **Anti-Goodhart:** APEX 8-submodel agreement on the top-50 active calls
is Gram+ **84.8%**, MDR **87.6%**, Gram- **95.7%** (not mean-gaming); 0% hemolytic per the independent
ESMC model; novel (median identity 0.62, max 0.73 rapidfuzz; the shipped Levenshtein novelty gate sees
max **0.48** << 0.80; 0 exact matches); ESMFold2 re-check folds all 100 into confident helices
(median pLDDT **0.704**, ~feat-020's 0.702 and well above a length-matched random control's 0.574;
helix 1.00; muH 0.30; **0/100** misfold flags); byte-reproducible (two runs md5-identical).

## Directed evolution against APEX overfits the oracle -- a held-out-submodel result (2026-09-28, feat-022)

**Question.** feat-019 found the generator's *sampling* distribution has a ~linear ceiling on diverse,
non-hemolytic Gram+/MDR hitters. Directed evolution (a genetic algorithm) *constructs* peptides by
mutating the best ones toward higher APEX reward, so in principle it is not bounded by that sampling
distribution. Can it beat feat-021? (`experiments/directed_evolution.py`, seeded from the feat-020
pool's best non-hemolytic peptides, run on the 7 free H100s -- per the "use the free GPUs for anything
that raises the chance of winning" directive.)

**The trap, and the guard.** Optimising against an oracle invites Goodhart: the GA can find sequences
that game APEX's weights rather than gain real potency. To *detect* that, the reward is computed on a
**train split of APEX's 8 ensemble submodels [0-4]**, while selection and reporting use the **held-out
submodels [5,6,7]** -- which never influence any decision. A real gain generalises to the held-out
split; an artefact does not. (ESMC selectivity, novelty and ESMFold2 -- all independent of APEX -- are
the outer guards; the submodel split is the inner one.)

**Result -- the guard fired.** Over 12 rounds the **train** Gram+ SR of the selected top-50 climbed
**0.60 -> 0.76** (spectacular, if you trusted it) while the **held-out** Gram+ stayed essentially flat
(**0.59 -> 0.66**, gap widening to ~0.10-0.14). Scored on the *same held-out submodels [5,6,7]*,
**feat-021 beats the GA on its strong categories** -- Gram+ **0.71 vs 0.66**, MDR **0.62 vs 0.59** --
and the GA leads only on Gram- (0.64 vs 0.59) with Broad a wash. So the GA's apparent full-ensemble
superiority is largely *overfitting*: no net win, and feat-021's Gram+/MDR strength is confirmed to
survive on held-out submodels (i.e. it is not itself oracle-gaming). Had we optimised **and** reported on
the full ensemble -- the naive setup -- we would have "seen" Gram+ 0.60 -> 0.76 and shipped overfit
peptides. The submodel cross-validation is exactly what prevented that. **feat-021 remains the
submission**; the GA archive (`experiments/cache/de.npz`) is kept for the record. (A secondary
observation -- Gram- *generalises* better than Gram+ under evolution, train-vs-holdout gap ~0.03 vs
~0.12, since the Gram- submodels agree 95.7% -- looked worth chasing, so we probed it directly.)

**Gram--augmentation probe (rejected, `experiments/gn_augment_probe.py`).** Gram- is feat-021's
weakest, highest-leverage category, and the GA's Gram- gains generalise -- so can the GA's robust
(non-hemolytic, novel, small train-holdout gap) Gram-specialists lift feat-021's floor? Tested
honestly: **select on the train submodels, report on the held-out ones** (leak-free both ways).
Augmenting feat-021's candidates with the robust GA set and re-selecting raises held-out Gram-
(0.586 -> 0.67) and the weakest-category floor (0.586 -> 0.60) -- but **trades Gram+** (0.71 -> 0.67)
and MDR (0.62 -> 0.60); the mean of the four activity categories barely moves (0.637 -> 0.652). A
*targeted* swap that keeps feat-021's Gram+ peptides and only fills the Gram- gap is worse: the GA
Gram-specialists are weak on Gram+/MDR, so as Gram- climbs (up to 0.75) MDR falls **below** the old
floor (to 0.53) -- there is no free lunch. The ~0.014 floor gain sits well inside APEX's oracle noise
(AUROC 0.62 on novel peptides, and the evolved peptides are further out-of-distribution than
generator samples, where APEX is *less* reliable), and it trades feat-021's strongest, most-reliable
category. **Not a defensible improvement -- feat-021 ships unchanged.** The disciplined read of the
whole directed-evolution line: evolving against this oracle buys nothing real over feat-021, and the
submodel cross-validation is what let us see that instead of shipping an overfit mirage.

## Maximin selection: raise the floor, but not by sacrificing the standout categories (2026-09-28, feat-023)

A fixed weighted sum (:func:`~amp_challenge_2027.oracle.balanced_success_score`) cannot maximise a
*minimum*. Since the five categories are ranked separately, the weakest is arithmetically the
highest-leverage number -- so we built a maximin top-list selector (``generate.select_maximin`` over
``oracle.category_rates``): greedily fill the assayed top-50 with the peptide strongest in whichever
category is currently weakest, held to the same non-hemolytic gate + < 80% novelty + 0.6 diversity
screens, byte-deterministic. On the shipped 400k pool it does exactly what it claims -- top-50 floor
**0.583 -> 0.627**:

| selection | Broad | Gram- | Gram+ | MDR | floor | mean-4 | Selectivity (median P, max P) |
|---|---|---|---|---|---|---|---|
| **score** (feat-021, shipped) | 0.644 | 0.583 | **0.750** | **0.667** | 0.583 | **0.661** | **0.004, 0.06** |
| maximin | 0.645 | **0.631** | 0.670 | 0.627 | **0.627** | 0.643 | 0.035, 0.49 |

But the floor is the **wrong objective for this competition**, and the numbers show why. The five
categories are ranked *separately* (`docs/COMPETITION.md`): there is no overall mean of category
scores, and advancement (top-20) is decided on **library** quality, not the category scores. What wins
is being a *standout* in individual categories -- and the shipped ``score`` selection dominates maximin
in **four of the five**: it ties Broad, and wins Gram+ (0.75 vs 0.67), MDR (0.67 vs 0.63) and
Selectivity (median predicted-hemolysis 0.004 vs 0.035, and maximin pushes one top-50 peptide to
P = 0.49, right at the gate -- eroding the rare 0%-hemolytic standout most AMP designs cannot match).
Maximin wins only Gram-, and has the lower mean-of-four. Trading three standout categories for one
contested Gram- gain is a bad deal when each category is its own leaderboard.

So maximin stays a documented, tested, byte-deterministic **alternative** (`--select maximin`) -- the
right tool only if the scoring were ever an overall mean-of-category-*ranks* (where lifting your worst
rank helps) or if one deliberately wanted a robust, no-weak-category entry -- while the shipped default
remains `--select score` (feat-021). The exercise did surface one competition fact worth recording: the
real assay panel is **15 Gram-negative + 5 Gram-positive** (75% Gram-), so the Broad category is
Gram--dominated and APEX's 7/4 Gram-/Gram+ bucket split *under*-weights Gram- relative to reality -- a
known oracle-transfer caveat (already flagged), not something to over-fit the selection to. Net across
feat-022 + feat-023: two serious, GPU-heavy attempts to beat feat-021 (construct better peptides;
re-balance the selection) both come back to feat-021 as the strongest *defensible* entry.

## Latest-SOTA generator research: why we keep the ReST-optimised AR-Transformer (2026-09-28)

Per the "research the latest/largest before committing" directive, we surveyed 2025-2026 AMP
generators before spending hours on a generator swap. Findings: **(1)** the organizers' own baseline is
**AMP-Diffusion** (ESM-2 latent diffusion + protein-LM embeddings, biorxiv 2024 -> Cell Biomaterials
2025) -- generate 50k, filter/rank with APEX, 46 synthesised, 76% inhibited bacteria incl. MDR at low
toxicity -- so reimplementing a big diffusion generator would reproduce the **excluded baseline**, not
an edge. **(2)** Our edge over that baseline is exactly the **optimisation layer it lacks**: ReST
fine-tuning of the generator toward the APEX hard-Success-Rate objective + an *independent* ESMC
selectivity gate + diversity/novelty screens. The baseline only *filters* APEX; we *optimise* against
it. **(3)** The newest generators (OmegAMP -- targeted, biologically-informed generation, arXiv 2025;
AMPGAN v3 -- agentic non-canonical AMPs, 2026; multi-modal contrastive diffusion) are alternative
*generators*, but the generator is **not our binding constraint** -- feat-019 showed the 0.6 diversity
screen and the broad-spectrum-vs-hemolysis biology are, so a larger generator reimplemented in the
remaining window is high risk for uncertain gain. **Decision:** keep the validated ReST-optimised
AR-Transformer; the differentiator is optimisation against APEX + selectivity, which is precisely what
the excluded baseline does not do. (Sources: AMP-Diffusion, Cell Biomaterials 2025 / biorxiv
2024.03.03.583201; OmegAMP arXiv 2504.17247; AMPGAN v3 arXiv 2606.17127.)

## Diverse out-of-the-box exploration on the free GPUs (2026-09-28, feat-024 + species/consensus/structure)

Four orthogonal, parallel bets to find any real edge over feat-021 (and to de-risk it). All predictions,
no wet-lab; feat-021 is the validated fallback so a null result costs nothing.

**feat-024 -- "all-rounder" ReST (rejected).** feat-020's gated ReST rewarded *pure* Gram+/MDR, so the
generator was never pushed toward broad-spectrum non-hemolytic ALL-ROUNDERS (strong on Gram-/Gram+/MDR
at once). We added the ranker's hard Gram- term to the ReST reward (`reward.balanced_reward` gains
`gn_w=0.75`, now pinned to the ranker by a parity test) and ran 4 gated rounds on the 8 H100s. It made
things WORSE, not positive-sum: over the rounds the reward-top-100 went Broad 58.3->55.2, **Gram+
43.2->35.8, MDR 42.7->39.0**, Gram- flat (66.9->66.3), reward 1.615->1.488. Rewarding Gram- just pulls
the generator toward Gram--specialists at the expense of Gram+/MDR -- the same trade, at the generator
level -- because true broad-spectrum non-hemolytic all-rounders are biologically rare (only ~5 clear
*all* Gram- species per 150k). Generator + submission left untouched.

**Species-level analysis -- the key finding (a real caveat, no clean fix).** APEX's Gram- bucket is
E. coli-heavy (3 of 7 columns), so the aggregate Gram- SR (0.58) HID the per-species truth: feat-021's
top-50 clears A. baumannii 1.00 and E. coli 1.00 but **K. pneumoniae 0.00 and P. aeruginosa 0.04** --
the two hardest GN ESKAPE species, both on the real 20-strain panel. So feat-021's *real*-panel Gram-
(and Broad, which is 75% Gram-) is likely below the oracle aggregate. Is it fixable? K. pneumoniae is a
near-hard limit (7 non-hemolytic clearers per 150k, APEX median MIC 322 uM). P. aeruginosa is reachable
(2318 clearers) and coverage can go 0.04->0.90 -- but the P. aeruginosa-killers are Gram--specialists
that crater Gram+ (0.75->0.31) and MDR (0.67->0.37); the portfolio curve has no sweet spot (w=0.6 buys
P. aeruginosa 0.35 only by dropping Gram+ to 0.585, MDR to 0.553). And chasing APEX's hard-species
predictions is the highest-Goodhart-risk move (APEX is least reliable exactly on the hard species). Net:
the hard-GN gap is a biological + oracle-transfer limit no achievable top-50 solves cleanly, so feat-021's
choice -- bet on the Gram+/MDR/Selectivity standouts it CAN win, plus the easy Gram- species -- stands,
now with this caveat explicit.

**Multi-predictor consensus -- independent confirmation + selectivity validated (reassuring null).**
Three independent AMP predictors (Macrel, amPEPpy, AI4AMP -- different architectures/data than APEX; a
real Keras-3 silent-failure bug in AI4AMP caught and fixed) were run on the top-50 + a 5k pool sample.
They CONFIRM the top-50 is genuinely AMP-like (every peptide called AMP by >=1 tool, 49/50 by >=2, 30/50
unanimous), extending the earlier Macrel-only check to four models. But they do NOT usefully re-rank:
correlation with APEX is ~0 (they are generic binary AMP classifiers; APEX is strain-specific MIC), and
an unweighted activity-consensus is a TRAP -- it drifts the top-50 to median P(hemolytic) **0.93 vs our
0.005**, because the generic classifiers reward the cationic/amphipathic signature that drives both
activity AND hemolysis. This directly confirms our hemolysis gate is doing real, correct work. Keep
APEX + the existing selection.

## Structure + HC50, and the one shippable hedge: a closed-form amphipathicity bonus (2026-09-28, feat-025)

The last two orthogonal probes, and the only optimisation that produced a shippable, positive change.

**Structure (ESMFold2-Fast on 2,646 peptides).** The shipped top-50 folds into confident, uniformly
helical structures -- **0/50 disordered-fold artifacts** (pLDDT median 0.711, above the random-peptide
0.574 floor; helix ~1.0 but non-discriminative at these lengths) -- confirming the selection is
well-folded, now at top-50 granularity. But the **Eisenberg hydrophobic moment** (muH, the amphipathic
membrane-disruption determinant) has median **0.311**, at the LOW EDGE of the canonical AMP band and
below every positive control (melittin 0.35, magainin 0.45, LL-37 0.88): the top-50 is adequately but
not *strongly* amphipathic, because `balanced_success_score` has no structure term. 12/50 sit below
muH 0.19 (nearly non-amphipathic), including APEX's single top-ranked pick (muH 0.105) -- a
mechanistic-plausibility flag, not a misfold. Crucially, **muH is a closed-form sequence function** (no
fold), so it can enter the byte-reproducible ranking while ESMFold2 stays offline.

**The amphipathicity bonus (feat-025, `--amphipathicity-bonus`, ADOPTED as the default at 0.2).** The
bonus adds `coef * clip((muH - 0.25) / 0.25, 0, 1)` to the APEX activity: a **smooth monotone floor-ramp**
that rewards any peptide above the weak-amphipathicity floor (muH 0.25) and saturates at 0.50. (An earlier
hard band [0.30, 0.65] was rejected in code review -- it excluded strongly-amphipathic peptides above 0.65
and had a discontinuous edge; the floor-ramp is monotone and never penalises more amphipathicity.) It is a
rare **orthogonal, positive-sum-ish hedge** rather than a category trade: within the ESMC-non-hemolytic set
muH and hemolysis are **uncorrelated** (Spearman +0.036), so boosting muH does NOT bring back hemolysis
(the gate already removed it), and a coefficient sweep on the 150k pool shows the hemolysis gate holds at
**0/50 predicted-hemolytic across every coefficient 0.0-0.4** while MDR stays flat and Gram- even improves
slightly at small coef.

Measured on the real 400k pipeline (bonus 0 vs 0.2, same seed): the top-50 muH median rises
**0.31 -> 0.40** (mean 0.31 -> 0.40) and the fraction of non-amphipathic picks (muH < 0.19, mechanistically
implausible "AMPs" that APEX likes for sequence reasons) drops **26% -> 12%**, at a category cost of Broad
-0.011, Gram- -0.014, **Gram+ -0.005** (our strongest category, barely touched), **MDR 0.000**, with 0/50
predicted-hemolytic held. Every category delta is smaller than APEX's 0.62-AUROC error bars -- i.e. the
"cost" is measured by the oracle we distrust, while the muH gain is oracle-independent, so transfer-adjusted
the change is net-positive. coef 0.2 is the knee: it captures a strong muH hedge before the GP cost
accelerates (0.3 costs GP -0.025, eroding our best category for marginal extra muH).

**This is the session's one clean, shippable improvement, and we adopt it as the shipped default** (the
organizers run the default `uv run generate`, so a winning config must BE the default). It deprioritises the
low-muH APEX picks that look like sequence-only artifacts and hedges APEX's transfer error at negligible,
oracle-internal cost, keeping every category standout and 0% predicted-hemolytic. muH is closed-form, so
byte-determinism holds. `--amphipathicity-bonus 0.0` recovers the exact feat-021 selection.

**HC50 safety window (a trained regressor).** We obtained real quantitative HC50 data (Rathore et al.,
*Commun. Biol.* 2025; DBAASP + Hemolytik, 1,926 peptides) and trained a ridge regressor on ESMC-600M
embeddings: held-out Spearman **0.674** / AUROC 0.864, beating the shipped binary classifier's 0.455 --
genuinely additive signal. The top-50's predicted **safety window (HC50/MIC50) is 5.8x** (median),
already near the JOINT activity+selectivity Pareto frontier: only 58 peptides in the 150k pool match the
top-50's activity floor AND its gates, at a mere +6% median safety window. An aggressive safety-window
objective is a bad trade (pure-SW crashes Gram+ 0.75->0.25, MDR 0.667->0.333); only a tiny blend
(kappa~0.01-0.02) is near-free (+7-15% window at ~0 Gram+/MDR cost). We do NOT adopt it: the gain is
marginal, the model is a v0, its training data is **GPLv3** (a licensing conflict with the MIT-only
submission), and most competitors are hemolytic so we are likely already winning Selectivity by a wide
margin. Keep the shipped selectivity as-is.

**Bottom line of the whole diverse-exploration phase.** Three independent orthogonal checks (multi-
predictor consensus, HC50 regressor, ESMFold2 structure) all confirm the feat-021 selection is sound;
four optimisation attempts (directed evolution, maximin, all-rounder ReST, safety-window) are rejected as
trades; one real caveat is documented (the hard-GN species gap, unfixable); the Phase-2 advancement gate
is measured with the organizers' own `seqme` framework and found strong (uniqueness 1.0, diversity 0.839,
novelty 1.0, FBD firmly AMP-like); and one clean, deterministic, selectivity-preserving hedge -- the
feat-025 amphipathicity floor-ramp -- is **adopted as the shipped default (coef 0.2)** because it improves
mechanistic transfer at oracle-internal-only cost. The generator, the balanced objective and the ESMC
selectivity gate are otherwise unchanged from feat-021.

## The Phase-2 advancement gate, measured with the organizers' own framework (`seqme`)

Every optimisation above targets Phase-3 category scores (the top-50 that gets assayed). But
**Phase 2 decides who even advances**: `docs/COMPETITION.md` says the full 50,000-sequence library is
ranked by `seqme` for **diversity, novelty against known-AMP databases, and physicochemical property
distributions**, and only the **top 20 advance**. Heavy top-50 tuning is moot if the library fails this
gate -- and until now we had never measured it the way the organizers will. So we installed `seqme`
(0.5.1, BSD-3-Clause, szczurek-lab -- a measurement tool, deliberately kept out of the submission's uv
env so it cannot bloat the byte-reproducible entry point) and scored the shipped feat-021 library
directly (`experiments/measure_library.py`).

**Diversity / novelty (the shipped 50k library).** `Uniqueness = 1.000` (all 50k distinct),
`Diversity = 0.839` (normalized pairwise Levenshtein -- high), `Novelty = 1.000` (no exact match to any
of the 39,448 known-AMP reference sequences), and 3-gram `Jaccard = 0.0019` vs that reference (near-zero
n-gram overlap). The library is maximally unique, highly diverse, and fully novel by both exact and
n-gram measures.

**Physicochemical distributions.** Cationic and amphipathic exactly as AMPs should be: net charge
4.7 +/- 3.2, isoelectric point 11.6, Gravy -0.54, Boman 2.0, and Eisenberg hydrophobic moment
**0.40 +/- 0.19** -- which independently matches our own `oracle.hydrophobic_moment` (a cross-validation
of the feat-025 implementation against modlamp/seqme). Note the library-wide muH median (~0.37) is
healthy; the low-amphipathicity concern from the ESMFold2 check is specifically the *top-50 selection*
pulling low-muH standouts, not a library-wide defect -- which is exactly the scope of the feat-025 bonus.

**FBD -- is the novelty "AMP-like" or "garbage"?** Novelty=1.0 by exact match is trivially easy (any
perturbation achieves it), so we measured Frechet Biological Distance in ESM2-650M embedding space
against real AMPs, with two controls for calibration.

**Correction (2026-09-29, adversarial STRATEGY audit).** The originally-recorded FBD **1.938** was measured
on the *feat-021* library, **before** feat-028/029 introduced mixed-temperature sampling. Re-measured on the
**shipped feat-033 library** (byte-identical to the submission, `06e30960`), the numbers are:

| set (vs real-AMP anchor, ESM2-650M) | FBD | Diversity | Novelty(exact) | near-exact >0.80 to known AMPs |
|---|---|---|---|---|
| real-AMP held-out half (positive floor) | 0.078 | — | — | — |
| **shipped feat-033 library** | **3.44** | 0.837 | 1.0 | **1.80%** |
| un-ReST base generator (baseline) | 2.36 | 0.865 | 1.0 | 4.87% |
| uniform-random peptides (negative ceiling) | 5.43 | — | — | — |

So the hotter temp-1.6 body **did** push the library further off-manifold than the old number implied (3.44,
~62% of the way to random, not 35%) -- an honest Phase-2 realism cost of the mixed-temperature Phase-3 gain
that had never been re-measured. **But the relative read is not a loss.** Against the naive un-ReST baseline
(the challenge's proposed competitor proxy), the base is modestly more diverse (0.865 vs 0.837) and lower-FBD
(2.36) -- yet that lower FBD is substantially **memorization**: the base near-copies known AMPs at **2.7× our
rate** (4.87% vs 1.80% within 0.80 identity; 1.53% vs 0.67% within 0.90), and *near-exact matching against
DBAASP/dbAMP/APD is an explicit Phase-2 criterion that penalises exactly that*. Our fine-tuning bought genuine
**novelty** (fewer near-copies) and **cationicity** (88% vs 64%, net charge 4.8 vs 1.5, appropriate for AMPs)
at the price of a higher FBD from **specialisation toward the active chemotype** -- not from degeneracy
(homopolymer/low-complexity rate is ~0.6% in both). Net: on the axes Phase-2 actually scores, the shipped
library and the base trade wins (base: diversity +0.028; ours: near-exact novelty 2.7× better) -- a wash, so
the library remains a **defensible** Phase-2 entry, and moving toward the base would sacrifice near-exact
novelty *and* the Phase-3 activity the fine-tuning exists for. We therefore do **not** touch the generator;
this is now a competitor-relative measurement, not just an absolute one. (`scratchpad/measure_shipped.log`,
`measure_base.log`, and the near-exact sample.)

## SOTA literature check (2025-2026): confirm the stack, don't swap it

Before finalising, we scanned the 2025-2026 literature (PubMed) for any method that would justify
destabilising the validated AMP-Diffusion + APEX + ESMC stack this close to the deadline. Outcome: a
devil's-advocate-then-confirm. No pipeline-swap SOTA is worth the risk, and the newer work independently
validates our design choices.

- **Newer generators are incremental, high-risk swaps.** CFlowAMP (ESM-2 + conditional flow matching;
  +39.8% on its own generation-success score, 18x faster than diffusion; doi:10.1016/j.jmgm.2026.109401)
  and a soft-prompt-tuned ProtGPT2 with a voting ensemble (doi:10.1038/s44386-026-00045-6) are both
  plausible generators, but swapping the generator would throw away a validated, byte-reproducible,
  selectivity-gated pipeline for an unvalidated internal-metric gain. Not worth it near a one-shot deadline.
- **Independent mechanistic validation of feat-025.** A 2026 study of the "Janus alpha-helix"
  (doi:10.1016/j.colsurfb.2026.116171) shows that stronger amphipathic *radial face-segregation* -- i.e. a
  higher hydrophobic moment, exactly what our muH bonus rewards -- promotes persistent bacterial-membrane
  pore formation *while maintaining low hemolytic activity*. That is precisely the mechanism, and precisely
  the 0%-hemolytic-held result, behind feat-025: the amphipathicity hedge is grounded in independent 2026
  biophysics, not just our oracle.
- **Our design matches the organizers' own stated priorities.** Two 2026 reviews co-authored by the
  assaying PI, Cesar de la Fuente-Nunez, frame AMP generative design around multiobjective *potency +
  toxicity + reproducible validation* (doi:10.1016/j.cbpa.2026.102685; doi:10.1016/j.chom.2026.06.007) --
  which is exactly the APEX-potency + ESMC-selectivity + byte-reproducible pipeline we built. A separate
  generative-model comparison (doi:10.1186/s13040-026-00558-w) independently validates our anti-Goodhart
  concern -- AMP predictors show strong property-specific biases, so evaluation must be tailored to the
  objective -- which is the exact reasoning behind adding a mechanistic (oracle-independent) hedge.

Net: the literature confirms the stack is well-aligned with 2026 SOTA and with the organizers' priorities,
and independently supports the one change we made (feat-025). No pipeline change is warranted.

## Robustness of the shipped top-50 under the Phase-3 random 25-draw

Phase 3 draws **25 of the top 50 at random** and assays them — deliberately, to reward models that are
*reliably* good rather than lucky in one sequence. So the question that matters is not just the top-50
mean but its **variance under the random draw**. Bootstrapping the 25-draw over the shipped feat-025
top-50 (10k resamples, APEX category rates + ESMC selectivity):

| category | E[SR] over the 25-draw | p10 | p50 | p90 |
|---|---|---|---|---|
| Broad-spectrum | 0.633 | 0.622 | 0.633 | 0.644 |
| Gram-negative | 0.569 | 0.554 | 0.571 | 0.583 |
| Gram-positive | 0.745 | 0.740 | 0.750 | 0.750 |
| MDR ESKAPE | 0.667 | 0.667 | 0.667 | 0.667 |

The draw-to-draw spread is **tiny** in every category (MDR is flat; Gram+ never drops below 0.74), i.e.
the top-50 is homogeneous and the entry carries **no lucky-tail dependence** — precisely what the random
draw is designed to test. **Gram-positive (0.745, always ≥0.74) and Optimal Selectivity are our most
reliable winning shots**: all 50 peptides are active on ≥1 strain with P(hemolytic) < 0.08, so *every*
possible 25-draw is uniformly non-hemolytic, and most competitors' cationic AMPs are hemolytic. **MDR
(0.667) is rock-solid**; **Gram-negative (0.569) is the known weak category** (the K. pneumoniae /
P. aeruginosa species ceiling documented above), rarely clearing 0.60. Net: the submission is robustly
good on four of five categories and differentiated on Selectivity, with one understood, unfixable weak spot.

## Large-pool anti-Goodhart exploration (feat-027): more compute, rigorously tested, rejected

Directive: use the 8 H100s to push for a breakthrough. The shipped pool is only 400k (8x); the rare
broad-spectrum all-rounders (clearing the hard GN species K. pneumoniae + P. aeruginosa while keeping
GP/MDR) are ~5/150k, so a **much larger pool** might surface enough of them to lift our weakest
categories (Gram-negative, Broad). We generated **3.98M** fresh candidates across all 8 GPUs (8 shards,
temperature 1.6, seeds 1000-1007) and GPU-scored them with the 8-submodel APEX ensemble.

**The apparent result looked like a breakthrough.** Replicating the shipped two-stage gate on the 4M
pool (activity + 0.2*amphipathicity, ESMC hemolysis gate on the top 20k, then top-50) gave a top-50
with **Gram- 0.686 (vs shipped 0.569), Broad 0.709 (vs 0.633)**, same GP 0.750 / MDR 0.667, **0%
predicted-hemolytic**, muH 0.526, and P. aeruginosa clear-rate 0.04 -> 0.72. A +0.12 Gram- / +0.08 Broad
gain at no selectivity cost -- exactly what we wanted.

**But the anti-Goodhart held-out-submodel test says it is mostly overfitting.** Selecting a top-50 from
a 4M pool is enormous selection pressure on APEX (0.62 AUROC on novel peptides). Using APEX's 8 submodels
as train/held-out splits (`bulk_permodel`): select the top-50 with a TRAIN split, then measure it on the
HELD-OUT split. Apples-to-apples on held-out submodels, the big-pool selection vs the shipped top-50:

| on HELD-OUT submodels | big-pool selection | shipped feat-025 | verdict |
|---|---|---|---|
| Gram-negative | 0.63-0.68 | 0.54-0.57 | partially genuine (+0.06..0.14) |
| Gram-positive | **0.37-0.61** | 0.62-0.70 | Goodhart collapse (-0.09..-0.25) |
| MDR | **0.39-0.57** | 0.58-0.64 | Goodhart collapse (-0.05..-0.19) |
| Broad | 0.57-0.62 | 0.57-0.62 | wash |

The shipped top-50 is **self-consistent across train/held-out splits** (robust); the big-pool aggressive
selection's GP and MDR **evaporate on held-out submodels** -- its full-ensemble GP 0.75 / MDR 0.667 were
inflated by the very submodels that selected it. So the "free" Gram-/Broad gain is really a **Goodhart-
inflated trade**: it sacrifices our strongest, most reliable categories (GP, MDR) for a partial Gram-
gain. This is the "P. aeruginosa killers crater Gram+/MDR" caveat, now proven with a held-out test.

**And there is no portfolio of genuine all-rounders to harvest.** Requiring a peptide to clear
K. pneumoniae AND P. aeruginosa AND >=50% GP AND >=50% MDR across **>=6 of 8 submodels** while non-
hemolytic leaves exactly **1** peptide in the whole 4M pool (`INLKAIARLAKKIL`, all 8 submodels agree,
P(hemolytic) 0.024). K. pneumoniae is the binding ceiling: only 210 / 20,000 top candidates clear it
robustly. One genuine all-rounder shifts a 50-peptide list's category mean by ~0.006 (negligible), and it
is not in the shipped seed-42 library, so it could only be added by hand-injection -- which would break
byte-determinism and the "no hand-picked sequences" disclosure. Not worth it.

**Conclusion: rejected; the shipped feat-025 default stands.** 10x the compute confirms the validated
default rather than beating it -- the apparent gain is APEX overfitting, and K. pneumoniae remains a real
biological/oracle ceiling. A clean devil's-advocate-then-revise: explore aggressively, test with the
held-out anti-Goodhart guard, and keep the robust entry. Null results cost nothing; the byte-deterministic
`uv run generate` was never touched.

## Optimal Selectivity is already maximized (clean-signal frontier check)

Optimal Selectivity (HC50/MIC50 safety window) is our most differentiated category (~0% hemolytic while
~70% of known AMPs are hemolytic). We re-checked, with CLEAN signals only (APEX potency + ESMC
P(hemolytic), no GPLv3 HC50 model), whether the shipped balanced top-50 leaves any safety window on the
table. Adding a potency tilt `kappa * (1/MIC50 among non-hemolytic)` to the ranking on the 150k pool:
`kappa` 0.5-2.0 moves the median window only 0.082 -> 0.085 (MIC50 12.1 -> 11.8) with categories flat,
and `kappa=4` finally widens it to 0.087 but by then trades GP 0.72 -> 0.705 and MDR 0.667 -> 0.647. So
the shipped top-50 already sits on the safety-window Pareto frontier -- it is potent (MIC50 ~12 uM),
100% active on >=1 strain, and ~0% hemolytic -- and there is no category-neutral headroom. This confirms
the earlier HC50-regressor conclusion using only MIT-clean signals: keep the shipped selection.

**Session equilibrium.** The two highest-leverage remaining levers are now both rigorously exhausted:
a 10x-larger pool Goodharts (feat-027), and the Selectivity window is already maximized. Combined with
the earlier rejected explorations (directed evolution, maximin, all-rounder ReST, safety-window blend)
and the confirmed Phase-2 gate, the validated feat-025 default sits at the robust Pareto frontier for
this generator+oracle stack. Further oracle-based optimisation is Goodhart-risky (bad for a one-shot
submission) or negligible; the binding constraints (K. pneumoniae biology, APEX's 0.62 novel-peptide
AUROC) are fundamental, not tuning gaps.

## feat-028: mixed-temperature sampling -- a genuine, cross-validated breakthrough on the weak categories

The large-pool study (feat-027) showed oracle-based *selection* is Goodhart-limited. So we turned to the
*generator distribution* itself. The shipped generator samples at temperature 1.6 -- chosen for library
diversity, with an (unverified) claim of "no top-50 activity cost". We tested that claim with a temperature
sweep (0.8-1.6, ~500k pool each, gated top-50):

| temp | Phase-2 Diversity | Phase-2 FBD | Phase-3 Gram- | Phase-3 Broad | GP / MDR | muH |
|---|---|---|---|---|---|---|
| 1.6 (shipped) | 0.839 | 1.94 | 0.569 | 0.633 | 0.745 / 0.667 | 0.40 |
| 1.2 | 0.801 | 3.46 | 0.617 | 0.665 | 0.750 / 0.667 | 0.52 |
| 1.0 | 0.769 | 4.22 | 0.640 | 0.680 | 0.750 / 0.667 | 0.54 |

The "no cost" claim was **false**: a cooler pool lifts the top-50 Gram- and Broad substantially (the
activity-tuned generator's high-probability modes are its most-active designs), GP/MDR hold, and it is
*more* amphipathic. Crucially this is **not Goodhart** -- the held-out-submodel cross-validation
(feat-027's guard) shows temp-1.0's held-out Gram- 0.66 vs temp-1.6's 0.57 and held-out GP/MDR **flat**
(not the large-pool collapse), because lowering temperature changes what the generator *proposes*, not how
hard we select. But a cool pool costs Phase-2: library diversity 0.839 -> 0.769 and FBD 1.94 -> 4.22 (the
library becomes peaked), and Phase-2 is the advancement gate.

**Mixed-temperature sampling gets both.** The pipeline builds the 50k library as `top-list + pool[:50k]`,
so the library body is just the pool and the top-100 is prepended. We therefore draw **two** pools from the
same checkpoint and rng: a cool `--top-temperature 1.0` pool that is *ranked* for the top-100 (its best
peptides sit on the high-activity modes), and a hot `--temperature 1.6` body that forms the diverse 50k
library. Only the ~100 selected peptides are cool; the 49.9k library body stays hot-diverse. Measured on
the full pipeline (`uv run generate`, mixed default):

| | Phase-2 Diversity | Phase-2 Novelty | Phase-3 Gram- | Phase-3 Broad | GP / MDR | hemolytic | muH |
|---|---|---|---|---|---|---|---|
| shipped temp-1.6 | 0.839 | 1.0 | 0.569 | 0.633 | 0.745 / 0.667 | 0/50 | 0.40 |
| **mixed (1.0 top / 1.6 lib)** | **0.837** | **1.0** | **0.640** | **0.680** | **0.750 / 0.667** | **0/50** | **0.537** |

Best of both: **library diversity preserved (0.837 vs 0.839), novelty 1.0**, while the top-50 gets the full
temp-1.0 gain (**Gram- +0.071, Broad +0.047**), GP/MDR held, still 0% predicted-hemolytic, and *more*
amphipathic. Per-submodel cross-validation of the *actual* mixed top-50 confirms it: every one of the 8
APEX submodels sees its Gram- at 0.59-0.67 (mean 0.63) vs the shipped top-50's 0.54-0.61 (mean 0.56) -- a
genuine gain, consistent across all submodels (spread 0.08), not concentrated in the ones that selected it.
This lifts our two weakest categories (Gram-negative, Broad-Spectrum) to be genuinely competitive without
touching our GP/MDR/Selectivity standouts or the Phase-2 advancement gate. Adopted as the shipped default
(`--top-temperature 1.0`); `--top-temperature 1.6` (== --temperature) recovers the single-pool feat-025 run.

**Library temperature is also near-optimal (don't raise it).** Mixed-temperature decouples the library
temperature from the top-50 quality, so we checked whether raising the library body's temperature would
improve Phase-2 further. It does not pay: diversity rises only marginally (1.6->1.8->2.0: 0.839 -> 0.846
-> 0.851) while FBD -- distributional AMP-realism -- degrades sharply (1.94 -> 3.95 at temp 2.0, i.e. 35%
-> 72% of the way to random). The Phase-2 screen rewards libraries that are diverse AND realistic, so
trading a large realism loss for +0.01 diversity is net-negative. Keep the library body at 1.6.

**Top-temperature optimum is 0.8 (feat-029 -- revises the earlier "keep 1.0" call).** The prior conclusion
here was that a cooler top pool "concentrates the top-50 onto fewer motifs -- a correlated-failure risk",
so 1.0 was kept. That reasoning does not survive scrutiny: (i) the screened top-50 self-identity is FLAT
across top-temperatures (0.465->0.476 from 1.0->0.8), (ii) the within-list diversity screen hard-caps any
two selected peptides at <60% identity, and (iii) only the *selected* top-50 ship and the random-25 is
drawn from those -- the peaking of the *rejected* pool body never reaches the submission. The pool does
peak (the diversity screen rejects 2274 near-dups at 0.9, 3918 at 0.8, 8214 at 0.7, 17829 at 0.6), but
that does not concentrate what ships. A rigorous frontier (0.6-1.0, `experiments/bulk_permodel.py`) then
shows 0.8 is a clean *interior* optimum -- see the feat-029 section below. feat-029 ships top-temperature 0.8.

## feat-029 — Top-temperature frontier: 0.8 is the genuine optimum (revises feat-028's top-temp 1.0)

**Context / why re-opened.** feat-028 adopted mixed-temperature sampling with the ranked top-100 pool
drawn at `--top-temperature 1.0`. A follow-up fine-tune (0.8/0.9) in the same session was *rejected*
("keep 1.0") on the grounds that a cooler pool peaks the candidate distribution (the diversity screen
rejected 3918 near-dups at 0.8 vs 1533 at 1.0), risking correlated failure under the random-25 Phase-3
draw. That reasoning does not survive the data: **top-50 internal self-identity is essentially constant
across top-temperatures (0.465→0.476 from 1.0→0.8)** — the maximin/diversity selection absorbs the pool
peaking, so the final top-50 is no less diverse. The rejection was re-examined (devil's-advocate-then-revise).

**Frontier (seed 42, all other params shipped; per-submodel via `experiments/bulk_permodel.py`).**

(nov-med/nov-max = median / max top-50 nearest-known-AMP identity vs the full 39,448-AMP reference; muH via closed-form hydrophobic moment; per-submodel means via `experiments/bulk_permodel.py`.)

| top-temp | Broad | GN | GP | MDR | muH | nov-med | nov-max | phemo-max | self-id |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 (feat-028) | 0.680 | 0.640 | 0.750 | 0.667 | 0.537 | 0.667 | 0.800 | 0.079 | 0.465 |
| 0.9 | 0.691 | 0.660 | 0.745 | 0.667 | 0.553 | 0.667 | 0.769 | 0.083 | 0.471 |
| **0.8** | **0.696** | **0.663** | **0.755** | **0.673** | **0.558** | 0.691 | **0.769** | **0.073** | 0.476 |
| 0.7 | 0.689 | 0.654 | 0.750 | 0.667 | 0.510 | 0.667 | 0.800 | 0.069 | 0.487 |
| 0.6 | 0.684 | 0.651 | 0.740 | 0.667 | 0.524 | 0.696 | 0.800 | 0.057 | 0.485 |

**Clean interior optimum at 0.8** — every headline axis (Broad/GN/GP/MDR) peaks at 0.8 and turns over
below it; muH (closed-form, oracle-independent) also peaks at 0.8. Novelty holds and the worst case
improves (identities vs the full 39,448-AMP reference): tt0.8's top-50 nearest-known-AMP max is 0.769
(top-100 0.788) versus the shipped tt1.0's 0.800 — tt0.8 pulls the least-novel peptide *further* from the
0.80 gate, with fewer high-identity peptides (2 vs 4 above 0.75); median nearest-known identity 0.69, well
within the rule. Not a monotone "keep cooling" — a coherent peak (0.7 and 0.6 regress on Broad/GN/muH and
push a peptide back to the 0.800 gate).

**Anti-Goodhart (held-out submodels).** tt0.8 beats the tt1.0 baseline on **GN 8/8** independent APEX
submodels (per-submodel GN mean 0.626→0.656, min 0.586→0.609) and **Broad 7/8** (mean 0.630→0.646).
Every independent judge agrees the gain is real — it is not an artifact of the ensemble-mean ranking
(contrast feat-027/feat-022, where held-out submodels went flat/collapsed).

**Phase-2 is untouched by construction.** Only the ranked top-100 pool uses `--top-temperature`; the 50k
library body is still sampled at `--temperature 1.6`. The library bodies for top-temps 0.6–1.0 are
**byte-identical beyond the prepended top-100** (verified via md5 of `tail -n +201`), so Phase-2
diversity/novelty/FBD are unchanged. Lowering the top-temperature is therefore pure Phase-3 upside.

**Seed robustness.** The argmax-over-sweep was confirmed not to be seed-42 noise. tt0.8 vs tt1.0,
per-submodel held-out means (`experiments/scratch seed_analyze.py`):

| seed | GN tt0.8 | GN tt1.0 | GN win | Broad tt0.8 | Broad tt1.0 | Broad win | muH tt0.8 | muH tt1.0 |
|---|---|---|---|---|---|---|---|---|
| 42 (ships) | 0.656 | 0.626 | 8/8 | 0.646 | 0.630 | 7/8 | 0.558 | 0.537 |
| 43 | 0.652 | 0.636 | 6/8 | 0.643 | 0.626 | 7/8 | 0.512 | 0.534 |
| 44 | 0.651 | 0.644 | 5/8 | 0.641 | 0.636 | 6/8 | 0.545 | 0.536 |

tt0.8 beats tt1.0 on Gram- AND Broad category means at all three seeds -> the effect is structural.
Honest caveats: the margin and per-submodel agreement shrink at 43/44 (GN 8/8->6/8->5/8) and muH is not
uniformly higher (seed 43 favours tt1.0), so the shipped seed-42 draw is a *favourable* one and the true
expected gain is "consistent but modest", not "large". It is a free, downside-free, directionally-robust
edge on the two weakest categories (Phase-2 untouched, novelty and selectivity improved), which is the
right profile to take on a one-shot ranked entry.

**Decision: adopt `--top-temperature 0.8`** as the shipped default (was 1.0). Net vs feat-028:
Gram- 0.640→0.663, Broad 0.680→0.696, GP 0.750→0.755, MDR 0.667→0.673, muH 0.537→0.558, phemo-max
0.079→0.073, novelty and top-50 diversity held, Phase-2 unchanged, byte-deterministic. Official
validator re-run on the pushed URL (all 8 checks incl. the ≤80% novelty gate and reproducibility).

## feat-030 — Lys/Arg composition lever for Gram- (wet-lab-validated signal, but REJECTED as a selection bonus)

A 29-agent adversarial audit (the only surviving proposal of 10) surfaced an **oracle-independent** lever for
the weakest category. On `data/experimental/mic.csv` (46 wet-lab MICs, the sole APEX-independent ground truth),
Gram-negative Success Rate correlates with **Lysine-richness**: Spearman Gram- vs R/(R+K) rho **−0.44 (p=0.003)**,
vs frac-K +0.46 (p=0.001); Gram+ is null (p=0.32, so Gram-specific); and R/(R+K) is **not confounded** with net
charge / hydrophobicity / length (p=0.13/0.23/0.14) — an independent compositional axis. All reproduced
independently (manual Spearman/Mann-Whitney; scipy absent from the env). Meanwhile APEX drove the shipped top-50
to **~82% Arg-dominant (median R/(R+K) 0.67)** — the *opposite* of what wet-lab favours. Crucially this Arg-bias
is **shared across all 8 APEX submodels**, so the held-out-submodel guard is blind to it (raising `--gn-weight`
to 2.0 stays Arg-dominant and "passes" 8/8 while chasing a biased target); only external data catches it.

**Attempt:** a closed-form `--lys-bonus` term added to the top-pool ranking activity (mirroring the amphipathicity
bonus), tested as a Lys-dominant-tail clip and as the more-faithful continuous K/(R+K) form, swept over gamma with
per-submodel do-no-harm + ESMC selectivity + novelty + poly-K-degeneracy gates.

**Result — REJECTED as a default:** the APEX-Arg-dominant pool lacks enough high-APEX-activity Lys-rich candidates
for a selection bonus to move the composition. At do-no-harm-safe gamma the top-50 **median R/(R+K) does not shift
(stays 0.667, still in the wet-lab low-Gram- regime)** — only frac-K nudges +0.02 (within noise). Forcing a real
shift needs a gamma that degrades MDR/muH and drives poly-K runs (tail γ0.8 also broke 0%-hemolytic). An offline
map hinted the continuous form could shift at γ0.2, but the full ESMC two-stage pipeline does not reproduce that
shift. So the Lys signal is **genuine wet-lab biology but not cleanly implementable as a ranking bonus** on this
oracle-selected pool without collateral damage. The lever would need generation-side conditioning (a larger,
higher-risk change), not selection re-ranking. **feat-029 (top-temperature 0.8, validated) stands unchanged.**
The methodological lesson — "all 8 submodels agreeing is necessary but not sufficient; a bias shared by the whole
oracle needs an independent check" — is the durable takeaway.

## feat-031 — Wet-lab Gram- de-bias, done right: Arg-EXCESS penalty (ADOPTED, revises feat-030)

feat-030 rejected the wet-lab Lys signal as un-implementable. feat-031 revisits it with (1) the decisive
non-circular validation feat-030 lacked, (2) a better functional form, and (3) primary-literature corroboration —
and **adopts it as the shipped default (`--lys-hedge 0.4`)** as a strict Pareto improvement on APEX plus a
robustness gain on the 75%-Gram- real panel.

**(1) The decisive test feat-030 never ran — is APEX right about Arg?** Scored all 46 wet-lab peptides through
APEX (`experiments/bulk_permodel.py`, GPU) and correlated APEX's *prediction* against the *real* MICs:
- APEX has **no predictive signal on novel peptides**: APEX-Gram- SR vs REAL Gram- SR Spearman **−0.13 (p=0.38)**;
  APEX-Broad vs REAL-Broad **−0.29 (p=0.047, significantly negative)**; APEX-Gram+ +0.13 (p=0.39).
- APEX is **significantly Arg-biased on Gram-**: residual (APEX_GN − REAL_GN) vs R/(R+K) Spearman **+0.41
  (p=0.003)** — the more Arg-rich, the more APEX *over-rates* Gram-. APEX even has the sign wrong (its own
  APEX-GN vs R/(R+K) is +0.18; reality is −0.40, p=0.004). Bootstrap (5000×) P(sign correct)=0.996/0.998,
  CIs exclude 0; leave-one-out keeps real-GN-vs-R/(R+K) in [−0.47,−0.34] (no single peptide drives it).
So feat-030's rejection was **circular**: it judged a *correction for APEX's Arg-bias* by *APEX's own biased score*.
The shipped feat-029 top-50 sits at **82% Arg-dominant (median R/(R+K) 0.667)** — concentrated in APEX's blind spot.

**(2) Why the penalty works where feat-030's bonus failed.** feat-030 added a Lys *bonus* (reward high-Lys); the
Arg-dominant pool has too few high-APEX Lys-*rich* peptides, so a do-no-harm-safe bonus couldn't move the median
(stayed 0.667). feat-031 subtracts an Arg-*excess* penalty `lys_hedge · max(0, R/(R+K) − 0.4)` (`oracle.arg_excess`,
mirroring the amphipathicity term; deterministic). Because APEX Gram- SR is **flat across R/(R+K) 0.2–0.8** and
blend peptides (R/(R+K)≈0.5) are abundant with *equal* APEX score, the penalty tie-breaks the extreme-Arg tail
toward those blends — shifting composition at ~zero APEX cost. It targets Arg *excess*, never rewards pure Lys.

**(3) Literature corroboration (4-agent scan, 5 citations PubMed-verified; verdict "ship milder").** Matched-peptide
studies support the direction in exactly the scored slices: Hackney 2026 (W6K8 Lys > W6R8 Arg — more bactericidal
*and* less hemolytic); Zou 2007 (Arg's edge "much more pronounced vs *S. aureus* than *E. coli*", shrinks at
physiological salt → a mild Lys tie-break is low-cost in the Gram-/physiological regime we score); Wang 2025 (Arg
helps Gram+, hurts Gram-, raises hemolysis). Refinements adopted: penalize Arg *excess* not pure Lys (best real
chemotype is a Lys/Arg *blend*, van der Walt 2025); keep it *tie-breaker magnitude* (R/(R+K) is second-order, the
signal is one n=46 family); *do not hurt Gram+* (Mishra 2016: Arg helps MRSA).

**Real-pipeline sweep (seed 42, all shipped defaults; APEX per-submodel via `experiments/bulk_permodel.py`, ESMC
selectivity + novelty as shipped).**

| `--lys-hedge` | Broad | Gram- | Gram+ (submodel-min) | MDR | med R/(R+K) | Arg-dom | med P(hemo) | novelty max |
|---|---|---|---|---|---|---|---|---|
| 0.0 (feat-029) | 0.696 | 0.663 | 0.755 (0.47) | 0.673 | 0.667 | 0.82 | 0.0031 | 0.769 |
| **0.4 (feat-031)** | **0.707** | **0.680** | 0.755 (0.48) | 0.673 | 0.600 | 0.70 | 0.0030 | 0.769 |
| 0.6 (too strong) | 0.700 | 0.671 | 0.750 | 0.667 | 0.600 | 0.62 | 0.0034 | — |

**h0.4 is a strict Pareto improvement:** APEX Broad +0.011 and Gram- +0.017 *rise*, Gram+/MDR hold *exactly*
(0.755/0.673 — it does not flip the Arg-heavy hard-hitters, i.e. genuine tie-breaker behaviour, protecting the
Gram+ standing the literature warned about), the Gram+ **submodel-min improves 0.47→0.48** (more robust, not
mean-gaming), and composition de-biases toward blends (Arg-dominance 0.82→0.70, median R/(R+K) 0.667→0.60). h0.6
begins flipping Gram+/MDR (too strong), so 0.4 is the knee. **Anti-Goodhart all pass:** 0%-ESMC-predicted-hemolytic
held (median P 0.0030, max 0.073); novelty held/improved (max 0.769 < 0.80, median 0.691→0.667); amphipathicity
held (muH median 0.514, 0% non-amphipathic); ESMFold2 re-fold of matched Lys-rich vs Arg-rich candidates shows
Lys-rich fold *as well or better* (pLDDT 0.730 vs 0.673, helix 1.00). Structure-as-independent-predictor was also
tested (pLDDT vs REAL Gram- rho +0.245, p=0.094) — a weak trend, **not adopted** (too noisy/degraded folder;
avoids a Goodhart-prone structural term). 140 tests pass. Predictions only — no wet-lab claim; the value is
reducing a *documented, quantified* oracle bias at zero measurable APEX cost, an edge from data no other team has.

**Seed-robustness (42/43/44), full pipeline.** The de-bias replicates at every seed and never costs a
category: h0.4 vs h0.0 APEX Δ = seed42 (Broad +0.011, GN +0.017, GP/MDR +0.000), seed43 (all +0.000 —
free), seed44 (Broad +0.007, GN +0.009, GP +0.005, MDR +0.000); composition consistently de-biased
(median R/(R+K) → 0.60, Arg-dominance 0.82→0.70 / 0.82→0.64 / 0.84→0.66). So h0.4 is ≥ h0.0 on all four
categories at all three seeds (improves at 42/44, neutral at 43, never worse) — the tie-breaker signature.
**Phase-2 non-regression (seqme, ESM2-650M):** h0.4 library = h0.0 to 5 dp — Uniqueness 1.0, Diversity
0.83673 (vs 0.83673), Novelty 1.0, 3-gram-Jaccard 0.00211, charge 4.82, muH 0.371 (only 100/50,000 =
0.2% of the library changes). **Adopted as the shipped default `--lys-hedge 0.4`** (feat-029's top-temp 0.8
and the ESMC gate otherwise unchanged). **Official validator PASSED** on the pushed commit `2f7bb3c` (fresh GitHub clone + `uv sync` + generate ×2,
all 8 checks incl. byte-identical reproducibility and the ≤80% novelty gate, real ESM++/ESMC path — *"All
checks passed. Submission is valid!"*; fresh-clone output library `9a3278c9` / top `61becbab`, byte-identical to local).

## feat-032 — Lys-conditioned Gram- ReST generator: EXPLORED then REVERTED (apparent "domination" was ensemble-Goodhart — see the REVERTED subsection at the end)

> ⚠️ **This section records the feat-032 exploration as it stood at adoption; it was subsequently REVERTED.**
> The "strict domination" below is measured in APEX currency and was shown to be an **ensemble-Goodhart illusion**
> (a matched APEX-only control ReST reproduced the same held-out gain *without* the wet-lab prior), and it regressed
> near-exact novelty ~2.6×. Read the **"feat-032 REVERTED"** subsection at the end for the disproof and the lesson;
> the shipped generator is **feat-031**.

feat-031 corrected APEX's Arg-bias at *selection* time. feat-032 pushes the same wet-lab signal to the
*generator*, raising the ceiling of the weakest category (Gram-) — and, unlike the rejected all-rounder ReST
(feat-024), it **generalises across held-out submodels** because its reward carries an *oracle-independent*
term.

**Method.** Rejection-sampling fine-tuning (`experiments/rest_finetune.py`, 8×H100): sample → APEX 8-GPU +
ESMC score → keep top-reward novel → low-LR fine-tune with a corpus anchor. The novel ingredient is a new
`--lys-w` reward term (`reward -= lys_w · oracle.arg_excess`, the same Arg-over-Lys-excess feat-031 penalises),
so the fine-tune distils toward the **Lys-rich Gram-actives APEX under-samples but the wet-lab data favours**.
Config: `gn_w 1.5` (Gram- target), `lys_w 0.6`, hard ESMC selectivity gate (`--hemo-gate 0.5 --hemo-lambda 0`),
3 rounds. A diagnostic first (sampled 1.89M at top-temp 0.8, scored per-submodel on GPU): the current generator
*already* produces abundant Lys-rich candidates (12.2%) and Lys-rich all-rounders scale linearly — so ReST
**refines the distribution rather than inventing a chemotype**, which is why it can generalise.

**Apparent result (later disproven — see the REVERTED subsection) — `best.pt` (round 3) *appeared to* strictly dominate feat-031 on the APEX axis (shipped top-50, seed 42):**

| | Broad | Gram- | Gram+ | MDR | submodel-min (B/GN/GP/MDR) | R/(R+K) | med P(hemo) | Phase-2 div |
|---|---|---|---|---|---|---|---|---|
| feat-031 | 0.707 | 0.680 | 0.755 | 0.673 | 0.585/0.609/0.470/0.520 | 0.60 | 0.0030 | 0.837 |
| **feat-032** | **0.751** | **0.749** | 0.755 | 0.673 | **0.645/0.666/0.565/0.553** | **0.50** | **0.0012** | 0.831 |

**Anti-Goodhart — triply supported (this is *not* feat-024):** (1) a **held-out submodel split** (select the
top-100 on submodels 0–5, measure on 6–7) gives Gram- **0.633 → 0.843** — the gain generalises to submodels the
reward never saw, whereas feat-024's all-rounder ReST was flat on this exact test; (2) the **submodel-min rises
on all four categories** (the worst-case ensemble member is stronger, the opposite of mean-gaming); (3) the
composition moved the **independently-validated wet-lab-favourable way** (R/(R+K) 0.60→0.50, aromatic 0.21→0.15);
(4) ESMFold2 re-fold of the shipped top-50: **0/50 misfold flags**, pLDDT 0.709, helix 1.00; (5) **byte-deterministic**
(two full `generate` runs md5-identical, library `7e3641fa` / top `de8ed8a2`).

**Phase-2 survives the diversity collapse.** ReST narrows the generator (pool self-diversity 0.36→0.12 at
top-temp 0.8), and the pipeline's <0.6 screen rejects **66,756** near-duplicates (vs feat-031's ~3–4k). But the
library **body** is sampled hot (temp 1.6), which recovers diversity: the shipped 50k library scores seqme
**Diversity 0.831 / Uniqueness 1.0 / Novelty 1.0** — statistically the same advancement-gate standing as feat-031
(0.837). A diversity-preserving variant (ReST-v2: anchor 0.5 + dedup-cap) also passes Phase-2 (div 0.842) but
gives a smaller held-out gain (GN 0.699), so `best.pt` is chosen. 140 tests pass.

**Adoption (subsequently REVERTED — see below).** Promoting `best.pt` → `checkpoint/generator.pt` was auto-mode-gated as a shared-resource change and
**user-authorised**; feat-031's generator is backed up at `experiments/rest/feat031_generator_backup/` for instant
rollback. Predictions only — no wet-lab claim. **Official validator PASSED** on the pushed commit `efa6784`
(fresh GitHub clone + `uv sync` + generate ×2, all 8 checks incl. byte-identical reproducibility + ≤80% novelty +
real ESM++/ESMC path — *"All checks passed. Submission is valid!"*; fresh-clone output library `7e3641fa` / top
`de8ed8a2`, byte-identical to local). The top-50 profile is seed-robust (identical at seeds 42/43/44).

### feat-032 REVERTED — the adversarial review caught an ensemble-Goodhart illusion (2026-09-29)

The feat-032 adoption above was **reverted** after a 4-agent adversarial review + first-hand verification
disproved its central claims. This is the devil's-advocate discipline working, and the lesson is important
enough to record in full.

**What was wrong with the "strict domination":**
1. **The APEX gains don't transfer.** APEX has ~0 correlation with real activity on novel peptides (feat-031's
   own decisive test: Gram- rho −0.13). So Broad 0.707→0.751 / GN 0.680→0.749 are gains in a currency that
   does not convert to the real (wet-lab) score — the axis on which the competition is actually judged.
2. **The held-out "generalisation" is mechanical, not real.** APEX's 8 "submodels" are hyperparameter variants
   of ONE model on ONE dataset (train[0-5]/holdout[6-7] correlation **r≈0.82** on Gram- log-MIC), so any reward
   gain transfers to the "held-out" block regardless of real activity. The **decisive matched control** — an
   APEX-only ReST (`gn_w=1.5, lys_w=0`, i.e. feat-032 minus *only* the wet-lab term) — reached held-out Gram-
   **0.829**, essentially equal to feat-032's 0.843, **without the wet-lab prior**. So the "wet-lab prior made
   it generalise where feat-024 didn't" story is false; the lift is a harder GN push through correlated
   submodels. The aggregate Gram- is also **easy-species inflation** (A. baumannii / E. coli saturate ~0.99,
   while the hard clinical *K. pneumoniae* clears just 0.022).
3. **A real near-exact novelty regression (~2.6×).** The Lys-conditioning ReST pulled the generator onto the
   known-AMP manifold: library sequences within 0.80 Levenshtein-identity of a known AMP rose **1.56% → 4.08%**
   (within 0.90: 0.52% → 1.32%), and it **regenerated Penetratin exactly**. COMPETITION.md Phase-2 does
   "exact and near-exact matching against DBAASP, dbAMP, APD" — so this is a genuine advancement-axis regression.

**Why revert (winning-maximising, one-shot):** feat-032's real edge over feat-031 is thin (composition
R/(R+K) 0.60→0.50 and selectivity 0.0030→0.0012, both already in the wet-lab-favourable blend zone) and
feat-031 **already banks that composition win via the safe selection route** — without a generator swap, without
the near-exact regression, and without the diversity collapse (pool 0.36→0.12). On a no-resubmission one-shot,
the clean, validated, novel feat-031 maximises the real chance of winning: clean Phase-2 advancement **and** a
real (if modest) top-50 win, versus feat-032's Goodhart-illusory gains bought with an unquantified advancement
risk. Checkpoint restored to feat-031 (`da70fb42`); the feat-032 checkpoint is kept at
`experiments/rest/lys_gramneg/best.pt` for the record.

**Methodological lesson (the durable takeaway):** the held-out-submodel guard we relied on **cannot detect
ensemble-wide Goodhart** — a bias shared by all 8 submodels (like APEX's Arg-bias, or here the joint over-rating
of easy/saturated species) transfers straight through a train/test split of correlated submodels. Two checks
are mandatory before trusting an oracle-optimised gain: **(a) a matched control that isolates the genuinely-new
term** (here, lys_w=0 showed the gain was not from the wet-lab prior), and **(b) at least one truly independent
axis** (wet-lab composition; near-exact novelty). The one lever that survived both this session is the wet-lab
composition prior applied at *selection* time (feat-031) — cheap, reversible, and independently grounded.

## feat-033 — Wet-lab-calibrated COMPOSITION ranking: rank within the APEX band, not by APEX (SHIPPED, supersedes feat-031)

**The question feat-031 left open.** feat-031 corrected *one* APEX bias (Arg-over-Lys on Gram−) with a small
`arg_excess` hedge, but still **ranked by APEX**. This session asked the sharper question: inside the
high-activity band our top-100 is actually drawn from, does APEX rank real activity at all — and if not, what
does? The test that answers it is the one no oracle-only pipeline runs: score every candidate ranking signal
against **real** activity.

**Diagnostic on the 46 wet-lab MICs (matched regime — generated peptides, the exact competition strains).**
Scoring the 46 through APEX and correlating each per-peptide signal against the real category Success Rates
(leave-peptide-out; `scratchpad/diag_oracle.py`, `diag2.py`, `sweep_ranker.py`):
- **APEX ANTI-ranks within the band.** Ranking the 46 by *any* APEX score (balanced / category / SR / min-MIC)
  gives a top-15 **real** broad Success Rate of 0.19–0.30 — at or *below* the 0.31 random-draw baseline. Every
  APEX per-peptide signal correlates *negatively* with real SR (Spearman −0.28..−0.31).
- **Composition predicts it.** frac_K +0.42 (broad) / +0.46 (Gram−); aromatic −0.31 / −0.32; argfrac −0.43 /
  −0.44 — all surviving partial controls for net charge, length and APEX score. Top-15 by `lys_fraction −
  0.5·aromatic` → real broad **0.46** (Gram− 0.54) vs 0.31 baseline; beats APEX on all four categories at
  top-15 and top-25. A *fitted* stack overfits n=46 (leave-peptide-out 0.41 < the fixed composition 0.47), so
  the ranker is kept **simple and fixed**, not learned.

**The apparent contradiction, and its resolution on 946 independent DBAASP peptides.** The strong n=46 effects
had to be checked against something bigger and independent, so a subagent reverse-engineered DBAASP's live API
(the SPA backend `https://dbaasp.org/peptides?targetSpecies.value=<sp>&limit&offset` + `/peptides/{id}`; the
documented `/api/v1/...` is dead — HTTP 200, empty body) and harvested **1,164 unmodified linear L-peptides /
6,254 strain-resolved ESKAPE MIC rows** (`scratchpad/dbaasp_mic.csv`), then ran APEX on all of them
(`dbaasp_*.py`). This **resolved** the tension — it is textbook **range restriction**, not a contradiction:
- Across the **full** DBAASP range, **APEX ranks real activity well** (Spearman **+0.33**; top-decile-by-APEX
  real broad 0.77, the best single ranker). So APEX is *not* generally broken — the n=46 "anti-rank" is because
  the 46 are all APEX-selected actives.
- Banded by continuous APEX potency, **within the APEX-high band APEX flattens**: full +0.33 → top-30% +0.11 →
  **top-15% +0.06**, while composition *rises* to **+0.265**. Our top-100 = the extreme APEX tail = this
  within-band regime, so composition wins *for us* — confirmed independently (n=142–284), matching the 46.
- The strong n=46 side-effects were partly inflated: DBAASP full-range says argfrac ≈ neutral (+0.05, not −0.43)
  and aromatic ≈ neutral, and **length** is the biggest full-range signal (+0.26). But *within* the APEX band,
  more Lysine weight monotonically raises real **broad** activity, and on DBAASP-within-band real
  **Gram-positive** too (0.60→0.71). **Honesty caveat (adversarial audit, 2026-09-29):** the Gram-positive
  half does NOT replicate on the matched 46 — pure composition's real Gram+ there is ≈ band-random (Spearman
  +0.12, top-7-by-comp real Gram+ 0.214 vs band 0.207). The shipped ranker *does* lift real Gram+/MDR on the
  46 (top-7 Broad 0.468 / Gram+ 0.357 / MDR 0.464 vs band 0.269/0.207/0.250), but Check A of the audit shows
  that gain comes from the **−phemo selectivity penalty** (−phemo alone → Gram+ 0.286), not the Lysine push.
  So we do not claim composition itself wins Gram+; APEX is kept only as the gate (its low Gram+ *prediction*
  is mis-scoring within the band, not a real loss), and the selectivity penalty — not composition — is what
  carries real Gram+/MDR in our regime. See the "feat-033 adversarial code review" section below.

**Calibration (what to ship, and what not to).** `scratchpad/dbaasp_within.py`, `dbaasp_hybrid.py`,
`bigpool_hybrid.py`:
- **fracK is the direction-consistent signal** — the one lever the matched 46 (+0.42) and independent DBAASP
  within-band (+0.27) agree on in *direction*. **length is not** (46: shorter better; DBAASP: longer better —
  a direct conflict), so length is left out. The aromatic penalty (0.5) is kept mild: neutral on DBAASP, helpful
  on the 46, and it improves selectivity.
- **A hybrid (rank-blend APEX + composition) is pointless.** With a hard `phemo<0.1` gate it barely moves
  APEX-Gram+ (0.25→0.28) and only *dilutes* the validated composition gain — because chasing APEX-Gram+ = chasing
  the Arg/hydrophobic (hemolytic) chemotype the real data reject. So APEX is used **only as the active-band gate**
  (top-20k by balanced Success Rate) and composition does the within-band ranking.
- **The Lysine push MUST be paired with the hemolysis penalty.** A composition-only top-50 hits ESMC
  P(hemolytic) **0.89**; `score = composition_score − 1.5·phemo` (within the gate) lands the shipped top-50 at
  **0% predicted-hemolytic** (median 0.001, max 0.038 — *better* than feat-031's 0.073).

**Shipped ranker** (`oracle.composition_score` + `model.ApexRanker._composition_rank`,
`generate.py --composition-weight 1.0 --aromatic-weight 0.5`, the new default; `--composition-weight 0` recovers
feat-031). Rank the top-100 by `lys_fraction − 0.5·aromatic_fraction − 1.5·P(hemolytic)` within the APEX-active
band. **This changes only the top-100 — the 50k library is byte-identical, so the Phase-2 advancement gate is
untouched.** feat-033a top-50: R/(R+K) 0.60→**0.33** (Lys>Arg in 14%→**94%**), aromatic 0.20→**0.10**; ESMC
P(hemolytic) max **0.038**; novelty 0/50 above 0.80 (top-50 median 0.69, top-100 max 0.80), near-exact clean
(0 above 0.90); within-list diversity clean; all 140 tests pass; compliance PASS; **byte-reproducible** (two
default runs → top `dc37c540…`, library `06e30960…`). The **official validator PASSED** on the pushed commit
`84e2b78` (fresh GitHub clone + `uv sync` + generate ×2, all 8 checks incl. the ≤0.80 novelty gate and
byte-identical reproducibility; fresh-clone output `06e30960`/`dc37c540`). **Seed-robust:** seeds 43/44 reproduce
the same top-50 composition (argfrac 0.33, frac_K 0.28, aromatic 0.11), selectivity (ESMC P(hemolytic) max <0.04,
0% hemolytic) and novelty (0/50 above 0.80) as seed 42 — not a seed-42 artifact.

**Honest reporting of the APEX numbers.** The top-50's APEX-predicted Success Rates are **Broad 0.64 / Gram-
0.85 / Gram+ 0.26 / MDR 0.33**. The very high Gram− and low Gram+/MDR *both* reflect APEX's chemotype bias, not
expected real activity; on **every** real-activity test (the 46 and the DBAASP within-band analysis) the
composition selection ties-or-beats APEX-potency selection on all four categories, including Gram+ and MDR. The
expected real gain is largest on the Gram-negative-dominated Broad category and the Selectivity window; we give
up APEX's (mis-calibrated) Gram+/MDR *predictions*, which the real data show do not track real activity inside
our band.

**Why this is the opposite of Goodhart.** feat-022/024/027/032 all failed because they optimised *against* the
oracle and the "gain" was oracle-internal. feat-033 does the reverse: it uses **real wet-lab + independent
external data** to *correct* the oracle's within-band ranking error, and the correction is validated on data the
oracle never touched. It is an explicit, reversible, one-shot-appropriate bet backed by two independent datasets
— an informational edge a pure oracle pipeline cannot find. Adopted as the shipped default with the participant's
explicit go-ahead; the actual Kaggle submission remains user-gated.

**"Why not just train a better oracle?" — we tried; it anti-transfers.** The obvious alternative to a
hand-picked composition signal is a *learned* oracle on the new real data. We trained an independent
ESMC-600M-embedding ridge regressor on the 1,164 DBAASP peptides' real broad-ESKAPE Success Rate
(`scratchpad/dbaasp_oracle.py`). It fits DBAASP well (5-fold CV Spearman **+0.56**) — but on the 46
held-out peptides (our generated-peptide regime) it **anti-transfers, Spearman −0.15**, worse than
random and no better than APEX (−0.28), while the simple composition score stays best (**+0.45**;
top-15 real broad SR 0.46 vs the oracle's 0.24). A sophisticated learned oracle overfits its training
distribution exactly as APEX does; the *reason* composition wins is that it is the one signal whose
direction is consistent across distributions (the matched 46 and the independent DBAASP within-band
both agree). This is the anti-Goodhart property, and it is why the shipped ranker is a two-term
composition score, not a model. It also closes the ranking search: APEX-ranking, a learned
ESMC-DBAASP oracle, a rank-blend hybrid, and a length term were each tested and rejected on real data;
composition is the winner.

## feat-033 adversarial code review — the selectivity λ, ground-truth-validated (2026-09-29)

A delegated adversarial code review of the shipped feat-033 raised 14 findings. The one load-bearing
*scientific* question (its #4): the hemolysis penalty λ=1.5 was calibrated on the balanced-activity
scale (feat-019) but reused unchanged on the ~5×-smaller composition scale, so *is the shipped
`comp − 1.5·phemo` ordering still the pure-`comp` ordering we validated against real activity?* We had
never checked λ against ground truth — only against the predicted-hemolytic %. So we did, on the actual
ESMC selectivity model, on our 238k generation pool **and** the 46 wet-lab + 946 DBAASP peptides within
the APEX-active band (`scratchpad/verify_lambda.py`, `scratchpad/decisive_lambda.py`).

**The penalty is a de-facto gate, so λ is not load-bearing.** On our pool the active band's phemo is
strongly **bimodal** — median 0.031 but 24% above 0.5. So `comp − λ·phemo` barely moves the near-zero
clean majority and strongly suppresses the confident-hemolytic tail: **λ = 0.5, 0.75 and 1.5 give a
near-identical top-50** (our pool: all clean, phemo max <0.08; the 46: real-SR 0.468 either way; DBAASP:
0.523–0.529). The review's "scale-mismatch" concern is therefore empirically **null** — there is no
continuum of mid-range phemo values for λ to mis-weight.

**The penalty is necessary, not cosmetic.** A pure-composition top-50 (λ=0) is measured **22%
ESMC-hemolytic, max P(hemolytic) 0.93** — confirming the `oracle.composition_score` docstring's 0.89
warning. Dropping the penalty would ship hemolytic candidates and forfeit the Optimal Selectivity
category (which needs a high HC50).

**On the matched regime the penalty does not cost potency; the apparent cost is a broad-regime artifact.**
Within the APEX band, `−phemo` *positively* predicts real activity on the 46 (our closest analog:
top-7 real SR **0.468** with the penalty vs **0.351** pure-comp) and *negatively* on DBAASP (the classic
potency–toxicity correlation across a much wider peptide range: pure-comp 0.769 vs 0.523). Pure comp's
higher DBAASP potency is unusable — it is bought by selecting the potent-**and-hemolytic** peptides, which
is matched-regime-worse and disqualifying for Selectivity. A hard phemo gate + pure comp was also tested
and does **not** beat the soft penalty (both ~0.52 on DBAASP; the gate cannot recover pure-comp's 0.77
because DBAASP's potent peptides *are* the hemolytic ones). **Conclusion: λ=1.5 is correct; no change to
the byte-validated shipped artifact.** Finding #4 was worth checking and it vindicated the current ranker.

**Hardening applied (all preserve the byte-identical default output `top dc37c540` / `lib 06e30960`).**
The review also surfaced real robustness/clarity gaps, now fixed and covered by new tests
(`tests/test_composition.py`, 15 cases — the default-on composition path previously had none):
- **No-selectivity-model fallback (#2):** if *both* the ESMC and physchem selectivity models fail to
  load, composition mode would have degraded to pure-Lys-maximisation — i.e. the 22%-hemolytic path
  above. `build_ranker` now falls back to APEX activity ranking (the validated feat-031 path) instead,
  with a clear warning.
- **Graceful gate (#1/#7):** out-of-band candidates were marked with a single `−1e9` sentinel, so if the
  novelty/diversity screens ever exhausted the band the overflow would be filled *lexicographically*.
  They are now ordered by APEX activity just below the in-band minimum, so any overflow degrades to the
  feat-031 next-most-active peptides, not junk — and the gate no longer relies on a magic constant.
- **Weight guard (#3):** `aromatic_weight` now clamps non-finite/negative values to 0.0 like its three
  sibling weights (a NaN/inf could otherwise silently destroy the ranking via `inf·0.0`).
- **Honest disclosure (#8):** composition mode prints a NOTE that it supersedes the enabled-by-default
  `--amphipathicity-bonus`/`--lys-hedge`; the stale README options table (#6, five wrong defaults) is
  corrected and the composition flags documented.
- **De-duplication (#9/#10):** the aromatic-residue set is now sourced once from `physchem`; the
  argsort-band + PLM-phemo block (previously hand-synced in three methods) is one shared helper.
Findings #11–#14 (micro-efficiency on the numpy ranking path, which is dwarfed by the APEX/ESMC passes)
were assessed and consciously deferred as churn on a validated one-shot artifact for no measurable gain.

## feat-033 adversarial STRATEGY audit — a byte-reproducibility DQ risk, caught and fixed (2026-09-29)

The code review above scrutinised the *implementation*. A separate 5-agent adversarial workflow then attacked
the *strategic* decisions (each agent tried to REFUTE one, a synthesis judge rated each). Four of five
self-refuted to "keep the status quo" — but one surfaced a genuine, previously-missed defect.

**Actionable flaw — the APEX sequential path was not byte-reproducible (FIXED).** `ApexScorer._run_pool` (the
parallel CPU path) pins `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` *because* the authors established that
multi-threaded oneDNN/MKL GRU reductions are not bit-reproducible — but `_run_apex` (the sequential path,
taken when `_resolve_workers→1`: small inputs, `device='cuda'`, or a low-memory/low-core grader where
`_auto_workers` returns 1) built its subprocess env WITHOUT those pins, and `APEX_predict.py` sets no threads
itself. So on any grader that took the sequential path, APEX ran multi-threaded → a boundary peptide could
flip at the 20k active-band edge → `top.fasta`/`library.fasta` bytes differ between the validator's two runs →
**check #8 fails = disqualification**, irreversible on a one-shot entry. Our DGX takes the *parallel* path
(44 workers), so our byte-repro passed and never exercised the hole; the prior 14-finding review missed it,
and the lone byte-repro test forces the parallel path. **Fix:** a single shared `_apex_env()` helper used by
both paths (root-cause, so the asymmetry cannot re-drift), plus a regression-guard test. Proven byte-neutral
on the shipped artifact — `_apex_env()` returns the identical env `_run_pool` built inline, and the parallel
path is our only path — so `dc37c540`/`06e30960` are preserved; and per `_run_pool`'s own invariant,
single-threaded APEX is identical regardless of sharding, so the now-pinned sequential path yields the *same*
bytes as the parallel path on any grader.

**Two "worth-testing" challenges — both confirmed the status quo on ground truth (`scratchpad/confirm_audit.py`).**
- *Per-category λ effect (the one un-run validation):* λ had only ever been scored against real BROAD SR. Broken
  out per category on the 46, the shipped `comp − 1.5·phemo` lifts **every** category above band mean — Broad
  0.468 / Gram+ 0.357 / MDR 0.464 vs band 0.269/0.207/0.250 — so the penalty does not hurt Gram+/MDR; it helps
  them. DBAASP's per-category potency "cost" is hemolytic-bought (pure-comp Gram+ 0.682 at ESMC phemo 0.589 vs
  shipped 0.477 at 0.013). λ=1.5 confirmed across all four potency categories, not just Broad.
- *A charge-patterning signal (muH) vs composition:* composition is permutation-invariant, so an amphipathic
  moment is a real un-benchmarked axis. But WITH the phemo gate applied, neither variant meets the
  pre-registered bar (tie Broad AND strictly beat Gram+ on BOTH datasets at ≤2% hemolytic): a `z(comp)+z(muH)`
  blend "wins" on DBAASP only by re-admitting **51.6% hemolytic** peptides (muH overlaps the hemolysis axis the
  gate removes) and *hurts* Gram+ on the matched 46 (0.357→0.321); a gated 70/30 split is matched-regime-worse
  (Gram+ 0.357→0.250). Composition stands.

**One honesty fix.** The "more Lysine → real Gram+ 0.60→0.71" claim is DBAASP-within-band only; on the matched
46 composition alone is ≈ band-random on Gram+, and the shipped ranker's real Gram+/MDR gain comes from the
−phemo penalty, not the Lysine push. The feat-033 write-up above is corrected to say so.

Net: the strategic thesis (composition ranking within an APEX gate, minus a selectivity penalty) is intact and
now validated per-category; the one concrete defect was a reproducibility hole, fixed byte-neutrally before the
one-shot submit.

## feat-035 — 24-hour deep validation: feat-033 confirmed near-optimal on the validatable boards, no artifact change (2026-09-30)

A full-day, GPU-backed, diverse-exploration pass (goal: maximize real winning odds, not incremental tuning).
Method: ground-truth-anchored (46 wet-lab MICs = AMP-Diffusion's own assayed set, i.e. NOVEL generated peptides;
1164 DBAASP MICs + ~570 DBAASP HC50 = NATURAL AMPs) + a 5-agent literature deep-research + offline GPU
experiments. All work offline in scratchpad; the validated artifact (`dc37c540`) was never touched. Outcome:
**no safe, robustly-supported, shippable improvement over feat-033 was found**; the session's value is rigorous
independent multi-angle validation for a one-shot entry, plus two user-actionable rule/eligibility flags.

**Optimal Selectivity validated on REAL HC50/MIC (never done before, and NOT circular — real labels, not APEX).**
On 109 DBAASP peptides carrying both real HC50 and real MIC, the shipped composition chemotype (top-20 by `comp`)
has median real safety window **92.7 vs 58.5 baseline** at equal potency (median MIC 1.6 µM), and real HC50 157
vs 122. Cationicity ranks the real safety window (+0.34), hydrophobicity destroys it (−0.39); `comp` itself is
weakly positive (+0.08) but its chemotype (Lys-rich, low-aromatic, low-hydrophobic) lands squarely in the safe
region. Raw `charge`/`cationic` rank SW marginally higher (108/94) but do NOT help real Gram−/Broad potency and
net-charge HURTS real Gram+ (−0.31) — a bad trade of a validated potency strength for a noisy selectivity nudge.
`scratchpad/sel_groundtruth.py`. This is our most winnable board and the modal APEX-first competitor ignores it.

**The APEX active-band GATE: rigorously tested keep-vs-drop → KEEP.** A per-category ground-truth decomposition
(`scratchpad/gt_percat.py`) reconfirmed composition predicts real Gram−/Broad (Spearman +0.48/+0.45 on the 46)
but NOT real Gram+ (+0.12, ns) — no feature does; literature agrees Gram+ is the harder class. Then: does the
APEX gate add real-activity signal beyond composition, or just exclude realistic high-comp peptides that are
really active? (`scratchpad/gate_value.py`, `bootstrap_gate.py`.) On the 46 (novel/OOD) the gate LOOKS harmful
(high-comp APEX-inactive realALL 0.513 > APEX-active 0.273) but the bootstrap gap is +0.275, **95% CI
[−0.02,+0.52] — includes 0, NOT robust** (n=46, cells of 6/17). On DBAASP the gate looks helpful (0.787 vs 0.577,
tight) but is **circular**: APEX-pathogen (Wan/de la Fuente, *Nat Microbiol* 2025 — the competition's OWN wet-lab
lab) is trained on public AMP DBs incl. DBAASP, so DBAASP is in-distribution/partially memorized for it, while
the 46 diffusion peptides are OOD (exactly our regime, where APEX ~0/anti-correlates with reality). Neither set
*robustly* supports dropping the gate; discarding it on a non-robust n=46 signal would be the feat-032 trap. The
hedged design (APEX gate + validated composition ranking) is correct under this irreducible uncertainty.
Corollary: composition is validated on BOTH sets (it is trained on nothing), so it — not APEX — is the real
ranking signal; every APEX-prediction-on-DBAASP number in this doc should be read as optimistic.

**Headroom is bounded; no generator lever is worth the one-shot risk.** Real activity rises monotonically with
composition (no saturation to comp 0.43/0.50) and the pool under-supplies it (feat-033 top-50 mean comp 0.24),
but the intersection high-comp ∩ realistic(not poly-K) ∩ novel(≤80%) ∩ APEX-in-band is THIN (`supply_realistic.py`:
~15 seqs at comp≥0.18, ~0 at ≥0.22 in a 114k pool) and removing the gate yields poly-K junk (100% low-complexity,
39/50 fail the novelty screen; `select_sweep.py`). Oversampling is the only SAFE lever but is marginal AND
low-shippability (thin tail + the shipped CPU-APEX path caps validator-safe pool size). ReST toward composition
is limited (its high-comp output is gated out unless it also chases APEX = Goodhart). No change pursued.

**Literature (5-agent deep-research, `scratchpad/research_notes/…`).** Independently corroborates the composition
thesis: TabPFN "coarse composition suffices" recovers ~96% of full-predictor performance and its advantage GROWS
with sequence distance (our novel-peptide regime); multiple sources find learned oracles overfit-in-distribution
and barely beat trivial baselines OOD; APEX's own successor (ApexOracle) admits APEX "struggles to generalize to
novel pathogens." Tools surfaced: **HemoPI2** (Rathore *Commun Biol* 2025; open; HC50 Pearson 0.739) as an
independent selectivity cross-check; **ApexAmphion** (de la Fuente, bioRxiv 2025; ProGen2+LoRA+PPO on a
MIC+physchem reward; 100/100 wet-lab active) as the theoretically-sound generator upgrade IF generation is ever
pursued — but its top failure mode is exactly reward-hacking a frozen proxy (Goodhart), reinforcing the feat-032
lesson. Competitive picture: ~15+ serious entries (top-20 contested); the modal competitor is APEX-first +
HemoPI2 gate → our composition ranking is differentiated where APEX fails, and Optimal Selectivity is the
least-contested board. **Decisive token-free test of the report's #1 idea** (`scratchpad/learned_vs_composition.py`):
a LEARNED model (LogReg/RF/GBM on 12 composition+physchem features, fit on 1164 DBAASP peptides) wins
in-distribution (DBAASP 5-fold Spearman 0.33–0.35 vs composition's 0.14) but **COLLAPSES out-of-distribution on
the 46 novel peptides (−0.11 / −0.03 / −0.05), while the hand-crafted `lys − 0.5·arom` transfers strongly
(+0.448)** — the anti-Goodhart thesis made quantitative: a learned ranker cannot beat the composition formula on
novel peptides (our regime), so feat-033's hand-crafted ranker is confirmed the right call and no learned/TabPFN
ranker is adopted.

**Two flags escalated to the participant (rule/eligibility, cannot be resolved autonomously).** (1) The
organizers' own materials disagree on the wet-lab draw — the site's How-It-Works section and the design PDF say
"25 from the **top 100**", the FAQ says "**top 50**". If it is top-100, list positions 51–100 are also assayed
and our composition tapers there (top-50 comp 0.24 → top-100 0.19); confirm with the organizers, hedge by keeping
the top-100 strong. (2) Registration reportedly requires an **institutional email** ("gmail not accepted") — the
entry uses a gmail account. Both need the participant to check with the organizers before the one-shot submit.

Recommendation: **ship feat-033 as-is.** It optimizes the three boards we can validate (Gram−, Broad,
Selectivity), does not sacrifice them chasing the two we cannot (Gram+, MDR), and is differentiated from the
APEX-first field. Full session detail: `scratchpad/SESSION_FINDINGS.md`.

## feat-035 code-review closeout — the selectivity λ is a GATE, not a miscalibration (quantified) (2026-09-30)

A background `/code-review` of the feat-033/034 delta (3 independent finder forks + a conventions pass +
empirical gate verification + full test runs) found **no surviving correctness bug** — all three correctness
analyses independently confirmed the composition/gate math and both refactors are behaviour-faithful, and the
suite passes. Its top finding questioned the shipped default directly, so it earned an empirical answer.

**Finding #1: does `λ=1.5` (documented as calibrated for `balanced_success_score`'s ~[0,2.5] scale) dominate
`composition_score` (std 0.062, spread 0.222) and secretly make the top-100 selectivity-ranked, not
composition-ranked?** Tested on the shipped band itself (`scratchpad/verify_lambda_bandphemo.npz` = the exact
`comp`/`phemo` for the 20 000 in-band candidates; analysis `scratchpad/codereview_finding1.md`):

- The band's `P(hemolytic)` is **strongly bimodal** — 46% < 0.02 (clean), 14% > 0.9 (very hemolytic), a sparse
  valley between. Band-wide, `1.5·phemo` (std 0.55) *does* out-swing `comp` (std 0.15), so the reviewer's
  mechanical point holds in isolation.
- **But the penalty acts as a selectivity GATE, not a fine-order term.** It pushes the ~54% hemolytic-mode band
  down and out, so the selected top-100 is **99% clean** (phemo std 0.0085, max 0.077). *Within* the top-100,
  `std(comp)=0.040` beats `std(1.5·phemo)=0.013`, and Spearman(final, comp)=**+0.90** vs (final, −phemo)=**+0.06**.
- **Triangulation settles it.** shipped top-100 ∩ composition-only top-100 = **69%**; shipped top-100 ∩
  selectivity-only (−phemo) top-100 = **0%** (same at top-50: 68% / 0%). The shipped list shares nothing with a
  selectivity ranking and most of its mass with a composition ranking → it is **gate-then-compose**, exactly the
  two-mode behaviour feat-034 recorded ("bimodal → de-facto gate"), now quantified — NOT "selectivity-primary
  with composition as a tiebreak."
- **The gate is nearly free.** Mean `comp` across a λ-sweep: 0.510 (λ=0) → 0.495 (λ=1.5) → 0.493 (λ=3.0). The
  clean mode is rich in high-composition peptides, so excluding the hemolytic mode costs ~3% relative composition.
- **The reviewer's suggested fix — rescale λ to the composition scale (~0.15) — would be harmful.** λ-sweep
  (top-100 by `comp − λ·phemo`): at λ≈0.10–0.15 the top-100 admits hemolytic peptides (max phemo 0.077 → 0.222,
  up to 0.92 at λ=0.10; %phemo>0.05 goes 1% → 15%) for **+0.011** composition. A large λ on a different scale is
  *deliberate and load-bearing*: it is the gate that clears the bimodal hemolytic mass; a "scale-matched" λ would
  turn the clean gate into a soft trade-off that ships hemolytic peptides — directly hurting the Optimal
  Selectivity board (our best). **No artifact change; no λ change.** The docs' "composition ranking within the
  APEX band, minus a selectivity penalty" is substantively correct; this addendum sharpens *penalty = selectivity
  gate* with the numbers.

**Findings #2/#3/#5 — byte-neutral honesty fixes (applied); shipped output unchanged.** The review's other
surviving items are non-default-path or maintainability issues that never touch the artifact: (#2) `build_ranker`
conflated "user set `--hemolysis-penalty 0`" with "selectivity model failed to load" — both leave `hemo=None`
and force the safe APEX fallback, but the message wrongly blamed an "unavailable" model; it now distinguishes the
two and says how to re-enable composition ranking. (#3) under `--select maximin` the `Ranking:` provenance line
named the composition score-ranker even though maximin ignores it; an explicit `SELECTION:` line now records the
method that actually ran. (#5) `lys_fraction`/`aromatic_fraction` already share the aromatic SET with `physchem`;
a new test also pins their fraction FORMULA to `physchem._frac`, so it cannot silently diverge. All three are
stdout- or test-only on paths the shipped default (`--composition-weight 1.0 --hemolysis-penalty 1.5 --select
score`) never executes; **158 tests pass** (156 + 2 new), and a 2× `generate` run on the patched code reproduced
the shipped artifact **byte-identical** (`dc37c540`/`06e30960`). #4 (compute composition on the band only — a
~1 s micro-opt on the hot path) and #6/#7 (defensive gate/branch complexity) were reviewed and left unchanged:
correct as-is and not worth re-opening frozen, byte-validated code. No λ, generator, checkpoint, or shipped-output
change.

## feat-035 — independent selectivity cross-check: a second learned hemolysis model also fails OOD (2026-09-30)

The deep-research report's #2 recommendation was to stand up an **independent** hemolysis model (HemoPI2,
Rathore *Commun Biol* 2025 — open SOTA HC50 regressor, RF-on-composition, R 0.739) as a non-ESMC cross-check
of the shipped top-100, since our own "0% hemolytic" verdict comes from the ESMC head we *selected* with.
HemoPI2's release model needs a Zenodo/sklearn-1.3.1 pickle + MERCI, so instead its published recipe was
faithfully reconstructed on **its own datasets** (`scratchpad/hemopi2_crosscheck.py`, `hemopi2_finding.md`):
RF on AAC-20 + 8 physchem, target −ln(HC50 µM), independent-set **Pearson R = 0.702** (paper 0.739) — a valid
HemoPI2-class proxy.

- **Apparent disagreement.** The independent model rates our top-100 median predicted HC50 **66.8 µM (10%
  <32 µM "hemolytic")** — mildly hemolytic, even ~ our own library body (75.2 µM), and far from known-safe
  peptides (149.9 µM). Our ESMC head had called the same top-100 **0% hemolytic** (max phemo 0.038). Two
  independent hemolysis models disagree about the entry.
- **The real-HC50 arbiter (1908 peptides) resolves it in our favour.** Our top-100's Lys fraction (p50 0.233,
  max 0.61) is **in-distribution** (real range p95 0.462, max 0.75); real HC50 rises **monotonically** with Lys
  (Spearman +0.224, matching HemoPI2's own reported +0.214: 64 → 101 → 101 → 199 → 175 µM across Lys bins), and
  real peptides in our top-100's Lys regime (≥0.20) have **median real HC50 133 µM, 65% >100 µM — genuinely
  safe.** That contradicts the RF's pessimistic call on our *novel* peptides.
- **Disambiguation nails it as OOD distribution shift.** At *matched* Lys, the RF reproduces reality for
  **natural** held-out peptides (lys[0.25,0.40): natural real 136, natural RF 114) but **under-calls our
  generated peptides ~40%** (same bin: our RF 69). The RF is well-calibrated on natural sequences and
  systematically pessimistic on novel generated ones — an **independent learned hemolysis model failing OOD on
  our peptides, exactly as APEX fails OOD on activity.**
- **Consequence.** This is a *third* independent confirmation of the session's central thesis (APEX, and now
  a HemoPI2-class model, are both OOD-unreliable on novel peptides; the ground-truth-anchored composition →
  safety relationship is what transfers), so the feat-033 selectivity thesis is **corroborated**. It also adds
  an honest caveat: our own ESMC "0% hemolytic" is *likewise* a learned-model OOD estimate — the two models
  bracket the truth, and the trustworthy basis for the selectivity claim is the **real-HC50-validated
  composition chemotype (SW 92.7; our regime 133–175 µM), not any single predictor's point estimate**. Do NOT
  ensemble HemoPI2 into selection: filtering on a model just shown OOD-unreliable would be the feat-032
  ensemble-Goodhart trap. **No artifact change.**

## feat-036 — amphipathicity (muH) tested as a potency lever, REJECTED on the OOD test (2026-09-30)

Profiled the shipped feat-033 top-100's structure (a documented gap — earlier ESMFold checks were on the old
APEX-ranked peptides, a different chemotype; `scratchpad/amphipathicity_finding.md`). The composition top-100 is
**charge-driven and only modestly amphipathic**: hydrophobic moment median **muH 0.29** (below every classic AMP
control — melittin 0.35, magainin2 0.46, LL-37 0.56), net charge ~7, net hydrophilic. Max-composition does *not*
sacrifice amphipathicity (within-top-100 Spearman(comp, muH) = +0.14) — the low muH is a chemotype property, and
it confirms the old feat-020 flag.

That raised a sharp question feat-035's blanket "hydrophobicity destroys the safety window" did not settle: is our
low muH leaving *safely-recoverable* real potency on the table? On real data **within our high-composition band**,
it looked like a genuine lever — muH → real Gram− SR **+0.192** (monotone) *and* muH → real HC50 **+0.131**
(positive! the highest-muH bin has the best HC50), because muH is the amphipathic *arrangement* of few hydrophobic
residues, not total hydrophobicity, so within a Lys-rich low-hydrophobicity context it adds potency without the
hemolysis cost. **But the decisive OOD test (the 46 novel peptides = our regime) rejects it:** muH → realGN/realALL
**collapses to +0.007 / +0.004** (vs +0.19 on DBAASP), while composition holds at +0.50 / +0.47, and a comp+muH
top-10 does not beat comp-only on real broad (0.455 vs 0.464). Amphipathicity is **another in-distribution mirage**
(like APEX, net_charge, HemoPI2) that does not transfer to novel peptides, so the low muH of our top-100 is **not a
deficiency to fix** — adopting a muH preference would be the feat-032 Goodhart trap. ~9th independent angle; all
converge on **ship feat-033**. No artifact change.

## feat-037 — Composition-envelope cap: keep the assayed top-100 inside the wet-lab-validated cationic envelope (SHIPPED, supersedes feat-033 as the generate default) (2026-09-30)

Every prior session since feat-033 closed on "ship feat-033 unchanged." feat-037 is the **first shipped-artifact
change since feat-033** — but a small, disciplined one that leaves feat-033's core intact and only removes a
now-identified extrapolation risk from the head of the list.

**What changed, and what did not.** feat-033's core — rank the top-100 by `lys_fraction − 0.5·aromatic_fraction −
1.5·P(hemolytic)` within the APEX active band — is **UNCHANGED and vindicated**. feat-037 adds one final selection
step: a **composition-envelope cap** (`--max-cationic-fraction 0.4445`, the new default) that demotes any candidate
whose cationic fraction `fcat = (K+R)/len` exceeds **4/9 ≈ 0.4444** — the maximum among the 46 wet-lab actives — below
*every* in-envelope candidate, so the assayed top-100 stays inside the validated envelope instead of extrapolating
past it. It is a re-ranking only: the **50k library body (records 101+) is byte-identical** to feat-033; only the
top-100 selection changes (53/100 top peptides differ, 31/50 in the top-50, because dropping the 16 over-envelope peptides cascades
through the 0.6 within-list diversity filter). New shipped hashes: top **`86639c72df6849b096dc91fbada0e0bb`**,
library **`a4153d03d98d9e568c4b3c8122bc15af`** (pre-cap feat-033 was top `dc37c540…` / library `06e30960…`,
recoverable with `--max-cationic-fraction 1.0`).

**The finding: the composition ranker extrapolated past the data it was calibrated on.** feat-033 was validated
where the 46 wet-lab actives live (fcat ≤ 4/9 ≈ 0.4444), but the unconstrained ranker pushed **16/100 of the top-100
strictly beyond that envelope**, concentrated at the HEAD of the list — **11 of the assayed top-50**, with positions 1–4
the most extreme (position 1 = `KKKLKKKLLKKKKKLRLL`, fcat 0.667), because "maximise Lysine-richness" runs off the end
of the calibrated region. On the 46 wet-lab MICs these beyond-envelope extrapolations map to **below-average real SR
on every board** — the declining right tail of the charge inverted-U (deep-research this session: Dathe 2001,
selectivity lost beyond ~+5 net charge; Jiang/Hodges 2008, therapeutic index peaks then falls; Malanovic/Lohner 2016,
lipoteichoic-acid sequestration explains why more-cationic can hurt Gram+ and why net_charge sign-flips OOD). The
library itself is healthy and broad; the Goodhart was in the *ranking*, so the fix is a *re-ranking*, not a library change.

**Evidence (honest kNN arbiter, k=5, LOO-validated on the 46 wet-lab MICs: broad +0.48 / gn +0.43 / gp +0.29 /
mdr +0.46).** Corrected 0.4445 top-50 expected Success Rate vs feat-033 (paired bootstrap, B=5000, k=5):

| Category | Δ Success Rate (capped − feat-033) | 95% CI | Robust? |
|---|---:|---|---|
| Broad | +0.013 | [−0.012, +0.039] (P>0 = 0.85) | directional, CI includes 0 |
| Gram− | +0.015 | [−0.019, +0.051] (P>0 = 0.80) | directional, CI includes 0 |
| Gram+ | +0.009 | [−0.007, +0.029] (P>0 = 0.85) | directional, CI includes 0 |
| MDR   | +0.011 | [−0.008, +0.032] (P>0 = 0.85) | directional, CI includes 0 |

All four are **directionally positive but none CI-robust** (every CI includes 0). The **direction** is robust, though:
feat-037 ≥ feat-033 on all four boards at **every k (3/5/7/9, top-50), never negative**.

> **Correction of an earlier bug.** The first version of this cap used `--max-cationic-fraction 0.444`, a truncation of
> the true ceiling 4/9 = 0.44444. The strict `>0.444` test wrongly demoted the ~8 peptides sitting *at* the validated
> ceiling (fcat = 4/9), which spuriously inflated the apparent gain to **Gram+ +0.029 / MDR +0.026 with 95% CIs
> excluding 0**. The corrected `0.4445` default keeps those ceiling peptides and demotes only strictly-more-cationic
> extrapolations; with them re-included the honest gain is the smaller, non-CI-robust set above. The value of feat-037
> is therefore **principled envelope-capping** — removing 16 extrapolations more cationic than *any* validated active,
> at no measured cost and with mechanistic support — **not a robust measured SR gain**.

**Mechanism:** the 16 removed peptides map on the 46 to **below-average** SR on every board — they were the weakest
members, so dropping them lifts the top-50 mean, if modestly.

**Threshold: 0.4445 (= the 4/9 wet-lab ceiling) is the right cap.** Tightening to **0.40** buys a little more Gram+ but
**loses the Gram− direction** and collapses within-list diversity — over-fitting the envelope. 0.4445 (just above the
literal wet-lab max 4/9) is the principled choice: it keeps peptides *at* the validated ceiling and removes only genuine
extrapolations beyond it.

**Corrected 0.4445 top-50 characterization** (all computational predictions; no wet-lab efficacy claim): `lys_fraction`
**0.282** (~unchanged vs feat-033), R/(R+K) **0.286** (Lys>Arg in **92%**), aromatic **0.100**, net charge **+7**,
length median **18 [13–20]**, `fcat` median **0.4118 / max 0.4444** (= the 4/9 ceiling; uncapped feat-033 max was
0.667), within-list diversity **58 clusters @0.6, largest 14/100** (vs feat-033 65/20 — comparable or better). The
in-pipeline **−1.5·P(hemolytic-ESMC)** selectivity gate and the **real-HC50-validated Lys-rich low-aromatic chemotype
are unchanged** (feat-035); the λ=1.5 selectivity gate is still applied.

**Honest caveats — this is a refinement, not a measured improvement.** The gain is a **kNN-model estimate on n=46**
(small). The removed peptides are **extrapolations with no direct wet-lab measurement** — the improvement rests on
the disciplined inference that beyond-envelope peptides behave like the validated cationic *edge* (where real Gram+
collapses), not magically better than it. The magnitude is **small (~+0.01 SR) and not CI-robust** (every CI includes
0); the value is **risk-management** — don't assay peptides more cationic than anything we have validated — not a
measured wet-lab improvement. Within-list diversity is comparable (largest ≥0.6 cluster 14 vs 20). This is an explicit,
ground-truth-supported, one-shot-appropriate refinement — it *removes* a known extrapolation risk rather than
chasing an oracle — and it keeps the standing rule that **all figures are computational predictions and we make no
wet-lab efficacy claim**. Tests: 7 cap cases in `tests/test_composition.py` (incl. 4/9-boundary keep-vs-demote,
env_min-relative offset weight-independence, ascending-fcat backfill order, `None` recovers the feat-033 order); full
suite green (165 passed, 3 skipped). Recovery: `--max-cationic-fraction
1.0` restores the pre-cap feat-033 selection (no peptide has `(K+R)/len > 1.0`, so the cap is a no-op);
`--composition-weight 0` still recovers the feat-031 APEX path.

**Also explored this session, not adopted (recorded for completeness).**
- **D1 + ProGen2 in-envelope diversification pools.** A masked-infill generator (ESM-C 600M, D1 ~46k) and **ProGen2**
  (from `anthropics/uplifting-biomolecular-modeling`, the one AMP-applicable kit; ~42k novel / in-envelope /
  diverse-from-both, 32k net-new 21–28mers) produced large *complementary* pools. But these are **floor-play material
  for a hedged library only** — under the competition's mean-over-random-draw scoring the expected score is set by the
  composition signal, not by generator choice; using them would require a risky Tier-2 library change; and their
  activity is **unproven** (no trusted oracle for novel peptides, per feat-035). **Not shipped.**
- **Hydrophobicity / aggregation penalty (S1 real-HC50 analysis + an independent AGGRESCAN cross-check — the two converge).**
  Penalizing aliphatic hydrophobicity (`frac_hydrophobic` over `{AILMFWVC}`, beyond the aromatic term; w ≈ 0.5) is a
  **real but trade-not-free** lever. S1's byte-faithful reconstruction (regenerated feat-037's exact 400k rank-pool,
  matched the shipped top-100 100/100) raises the **real-HC50 safety window ~1.1–1.4×** (P 0.80–0.90, directional, not
  CI-robust), and on the kNN-MIC activity arbiter it lifts **Gram+ (+0.014, P=0.90 — our weakest board, the first lever
  to move it)**, but it **costs Gram− (−0.017) and Broad (−0.006)** (our stronger activity boards) and within-list
  diversity (largest ≥0.6 cluster 23 vs 14); at w=1.0 it clearly regresses (Gram− −0.043, P=0.05). Because one ranking
  serves all five **separately-ranked** per-board leaderboards, trading a strong board (Gram−) for a weak-board Gram+
  gain plus already-strong Selectivity is not a clear win — feat-037, by contrast, is directionally ≥ feat-033 on **all**
  boards (no trade). Consistent with S1's own verdict (the decisive selectivity headroom is **upstream in the generator**,
  not in re-ranking) and with the aggregation premise test (non-robust at the realistic top-50-of-20k selection ratio).
  **Not shipped** (`physchem._HYDROPHOBIC` + `_frac` are present if revisited).
- **Upstream generator ReST toward low-hydrophobicity (feat-039-explored — NO-GO).** Directly tested S1's "the
  decisive lever is upstream in the generator" hypothesis: a 3-round rejection-sampling fine-tune (`scratchpad/rest_lohyd.py`,
  reward = composition + a hydrophobicity hinge, hard envelope gate, **no APEX/ESMC in the reward** so no feat-032
  ensemble-Goodhart) with a **matched control** (identical minus the low-hydrophobicity term). **NO-GO on three grounds:**
  (1) *fails the matched control* — composition steering (Lys↑, aromatic↓) inherently lowers `frac_hydrophobic` (K
  displaces hydrophobics; F/W *are* hydrophobic), so the control reached the same/**lower** hydrophobicity (fh 0.333 vs
  candidate 0.400) *without* the low-hydro term; the explicit term adds nothing CI-robust beyond composition. (2) *fails
  "robustly beats feat-037"* — a safety-for-activity **trade**: both candidate and control over-extrapolate to top-50
  lys 0.41–0.43 (past feat-037's validated 0.28) and **lose the activity boards** (kNN-MIC gp −0.033 / mdr −0.031;
  DBAASP-LOOSE broad SR **−0.267, CI-robust-negative**, potency 1.79× worse) for a safety gain that is non-CI-robust,
  inconsistent across DBAASP STRICT (2.07×) / LOOSE (0.71×), and partly circular. (3) Phase-2 **preserved** (diversity
  0.839, novelty 0.998, near-exact known-AMP 1.13% — *better* than the shipped 1.67% and far below feat-032's failed
  4.08%; the composition reward avoided the feat-032 near-exact-regeneration trap) — necessary but insufficient. **New
  durable insight (strengthens feat-037):** low-hydrophobicity is *downstream* of composition, not an independent lever;
  **generation-time enrichment** of a good-direction feature lets the composition ranker **overshoot** the wet-lab-validated
  envelope on the *un-capped* axis (lys 0.42 vs validated 0.28, where the activity boards degrade), whereas feat-037's
  **selection-time cap** stops at the validated composition — cheaper, reversible, and with no whole-library Phase-2 risk.
  Checkpoint `generator.pt` (`da70fb42`) unchanged; artifacts in `scratchpad/` (`rest_lohyd.py`, `select_gen.py`,
  `rest_cand/`, `rest_ctrl/`).
- **D3 — Gram+ physchem lever search.** An MC-corrected 16-feature battery found **no transferable Gram+ physchem
  lever**: every DBAASP signal sign-flips OOD, reconfirming feat-035/036 (net_charge reverses sign, amphipathicity is
  an in-distribution mirage). There is no safe knob to raise real Gram+ beyond removing the extrapolations, which
  feat-037 does.

**Validator status.** The official validator **PASSED** on a fresh clone of the pushed feat-037 commit
(`scripts/verify_submission.py <repo-url>`, `da7ae56`; artifact byte-identical at HEAD): **"All checks passed.
Submission is valid!"** — all 8 checks incl. the ≤80% novelty gate and generate-×2 byte-reproducibility; the
fresh-clone output is byte-identical to the shipped artifact (top `86639c72` / lib `a4153d03`), the 3rd independent
clean reproduction. Everything else about the entry (library body, generator, checkpoints, selectivity model,
compliance) is unchanged from feat-033.
