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
~0.12 -- is noted but not acted on, since it yields no clean full-ensemble win over feat-021.)
