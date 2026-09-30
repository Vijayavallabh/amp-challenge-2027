# Session Progress Log

## Current State

> **feat-037 (2026-09-30) — COMPOSITION-ENVELOPE CAP; SHIPPED, supersedes feat-033 as the `uv run generate` default.**
> The first shipped-artifact change since feat-033, and a disciplined one. feat-033's core (rank the top-100 by
> lysine_fraction within the APEX active band) is **UNCHANGED and vindicated**; feat-037 adds a final
> **composition-envelope cap** (`--max-cationic-fraction 0.444`, the new default) that demotes any top candidate
> whose cationic fraction fcat=(K+R)/len exceeds 0.444 — the max among the 46 wet-lab actives — below every
> in-envelope one, keeping the assayed top-100 inside the validated envelope. **50k library body byte-identical;**
> only the top-100 selection changes (63/100 differ, as dropping the 20 over-envelope cascades through the 0.6
> diversity filter). **NEW hashes: top `21fd02b7aa928f32c1ac6f6aeb1faa2b`, library
> `bba245dccc21be693a80c1b114748bb5`** (pre-cap feat-033 `dc37c540`/`06e30960`, recoverable via
> `--max-cationic-fraction 1.0`; `--composition-weight 0` still recovers the feat-031 APEX path). **WHY:** the
> uncapped composition ranker extrapolated — ~20/100 of the top-100 sat past the envelope (fcat>0.444),
> concentrated at the head (13 of the top-50; pos 1 = KKKLKKKLLKKKKKLRLL, fcat 0.667), ~6× enriched vs the
> library; on the 46 wet-lab MICs that region is where **real Gram+ activity collapses** (fcat>0.40 → Gram+ SR
> ~0.07 vs peak ~0.36). The library is healthy — the Goodhart was in the RANKING — so the fix is a re-ranking.
> **EVIDENCE (honest kNN arbiter on the 46, LOO-validated: broad +0.48/gn +0.43/gp +0.29/mdr +0.46):** capped
> top-50 expected Success Rate vs feat-033 (paired bootstrap B=5000) — **Gram+ +0.029 (95% CI [+0.003,+0.047])**
> and **MDR +0.026 (CI [+0.001,+0.054])** EXCLUDE 0 (robust); Gram− +0.023 (P>0=0.84) and Broad +0.022
> (P>0=0.91) are directional (CI includes 0). Robust at k=5 AND k=7 for the top-50, positive at all k (3/5/7) and
> both pools, never negative; top-100/k=3 positive but not CI-robust. Mechanism: the 20 removed peptides are
> below-average SR on every board, most on Gram+ (0.177 vs the 0.242 top-50 mean). Threshold 0.444 is optimal
> (cap-0.40 loses the Gram− gain + collapses diversity to 41/100). **Capped top-50 (computational predictions, no
> wet-lab claim):** lys_fraction 0.278 (UNCHANGED), R/(R+K) 0.286, aromatic 0.103, net +7, len 18, fcat median
> 0.400/max 0.438 (was 0.667), muH 0.331, Lys>Arg 88%, **0% predicted-hemolytic** (ESMC median 0.001, max 0.052
> top-50 / 0.075 top-100, all <0.5; max rose 0.038→0.052, real-HC50-validated chemotype unchanged). **3 new cap
> tests** in tests/test_composition.py (demote-over-cationic / in-envelope-unchanged / None-recovers-feat033-order);
> suite green. **HONEST CAVEAT:** the gain is a kNN-MODEL estimate on n=46; the removed peptides are
> EXTRAPOLATIONS with no direct wet-lab measurement (magnitude modest ~+0.02–0.03 SR; diversity marginally lower,
> largest ≥0.6 cluster 24 vs 20). An explicit, ground-truth-supported, one-shot-appropriate refinement, NOT a
> measured wet-lab improvement — the standing "all figures are computational predictions" caveat holds.
> **VALIDATOR:** top.fasta changed → the official validator MUST be re-run on the FINAL pushed commit (PENDING,
> not yet claimed passed); local 2× byte-repro re-confirmation pending. **Also explored, NOT shipped:** D1 (ESM-C
> 600M masked-infill ~46k) + ProGen2 (from anthropics/uplifting-biomolecular-modeling, the one AMP-applicable
> kit; ~42k in-envelope/novel/diverse, 32k net-new 21–28mers) diversification pools — floor-play material for a
> hedged library only (expected score set by composition, not generator choice; needs a risky Tier-2 library
> change; activity unproven), and D3 — an MC-corrected 16-feature battery found NO transferable Gram+ physchem
> lever (every DBAASP signal sign-flips OOD; confirms feat-035/036). Detail: `docs/RESEARCH.md` "feat-037". Both
> feat-035 user flags remain SET ASIDE per the participant.
>
> **feat-036 (2026-09-30) — CODE-REVIEW CLOSEOUT + INDEPENDENT CROSS-CHECKS + BYTE-NEUTRAL HARDENING; NO shipped-output change.**
> A same-day follow-up to feat-035. A background `/code-review` found **no surviving correctness bug** (156 tests). Its
> top finding + two more independent, ground-truth-anchored experiments all reconfirm feat-033; the shipped artifact is
> **byte-unchanged** (2× `generate` on the patched code → `dc37c540`/`06e30960`), now **158 tests**; the **official
> validator was re-run on the latest commit `a7ed06f` → all 8 checks PASSED, fresh-clone byte-identical**. (1) **The selectivity
> λ=1.5 is a GATE, not a miscalibration:** on the shipped band `phemo` is bimodal, λ clears the ~54% hemolytic mode, and
> the top-100 is 99% clean and **composition-ordered** (Spearman(final,comp)=+0.90 vs (final,−phemo)=+0.06; overlap with a
> selectivity-only ranking = 0%). The review's "rescale λ to the composition scale" would re-admit hemolytic peptides for
> ~0 composition gain — **no λ change.** (2) **Byte-neutral honesty fixes** (#2/#3/#5: penalty-disabled vs model-unavailable
> message; explicit `SELECTION:` line under `--select maximin`; a `physchem._frac` formula-pinning test) — stdout/test-only
> on paths the default never runs. (3) **Independent HemoPI2 selectivity cross-check** (reconstructed RF, R 0.702 vs paper
> 0.739) disagreed with our ESMC head on the top-100, but the **real-HC50 arbiter** resolves it in our favour (our Lys
> regime is in-distribution and safe, median real HC50 133 µM); the disagreement is generated-peptide OOD — a *third* learned
> model (after APEX, ESMC) failing OOD on novel peptides, while composition-on-real-data transfers. (4) **No safe Gram+
> lever:** no physchem feature specifically predicts real Gram+; net_charge helps Gram− more and *reverses sign OOD* (+0.22
> natural vs −0.31 on the 46 novel), and our top-100 is already well-charged — chasing Gram+ would backfire. Honest caveat
> recorded: "top-50 0% hemolytic" is an ESMC learned-model OOD estimate; trust the real-HC50-validated chemotype, not a
> per-peptide guarantee. Detail: `docs/RESEARCH.md` (code-review closeout + selectivity cross-check); `scratchpad/codereview_finding1.md`,
> `hemopi2_finding.md`, `grampos_finding.md`. **Recommendation unchanged: ship feat-033 as-is.** Both feat-035 user flags remain open.
>
> **feat-035 (2026-09-30) — 24-HOUR DEEP VALIDATION; feat-033 confirmed near-optimal; NO artifact change.**
> A full-day GPU-backed pass (ground truth + a 5-agent literature deep-research + offline GPU experiments; the
> shipped artifact `dc37c540` untouched) found **no safe, robustly-supported, shippable improvement** over feat-033
> and instead re-validated it from new angles. **Optimal Selectivity validated on REAL HC50/MIC** (109 DBAASP
> peptides with both, real labels not APEX → not circular: composition chemotype top-20 median real safety window
> **92.7 vs 58.5 baseline** at equal potency — our best, least-contested board). **APEX-gate keep-vs-drop tested and
> resolved to KEEP:** on the 46 (novel/OOD) the gate looks harmful but the bootstrap gap CI **[−0.02,+0.52] includes
> 0** (not robust, n=46); on DBAASP it looks helpful but is **circular** (APEX = the de la Fuente competition lab,
> trained on DBAASP-like data → DBAASP in-distribution). Dropping it would be the feat-032 trap. **No validated Gram+
> ranking signal exists** (composition +0.12 ns; literature agrees Gram+ is harder) and MDR is unvalidatable — so
> feat-033 correctly optimizes the validatable boards (Gram−/Broad/Selectivity) without sacrificing them. Headroom
> is bounded (oversample = marginal + CPU-APEX-runtime-capped; ReST toward composition = Goodhart-limited by the
> gate). Literature corroborates the composition thesis ("coarse composition suffices"; advantage grows with
> sequence distance) and the modal competitor is APEX-first → we're differentiated on Selectivity. **TWO USER FLAGS
> to confirm with the organizers before the one-shot submit:** (1) their OWN materials DISAGREE on the wet-lab draw
> — site/design-PDF say "25 from **top-100**", FAQ says "**top-50**"; if top-100, list positions 51–100 are also
> assayed (our composition tapers there). (2) registration reportedly requires an **institutional email** ("gmail
> not accepted") — the entry uses a gmail account. Detail: `docs/RESEARCH.md` "feat-035" + `scratchpad/SESSION_FINDINGS.md`.
> **Recommendation: ship feat-033 as-is.**
>
> **SHIPPED SUBMISSION — feat-033 (wet-lab-calibrated COMPOSITION ranking; current truth; feat-021→032 historical below).**
> `uv run generate` defaults to **`--composition-weight 1.0 --aromatic-weight 0.5`** on the feat-020/021 generator
> (`checkpoint/generator.pt` = `da70fb42`, unchanged). It ranks the top-100 by
> `lys_fraction − 0.5·aromatic_fraction − 1.5·P(hemolytic)` **within an APEX-active-band gate**, NOT by APEX
> potency. Rationale (docs/RESEARCH.md feat-033): built the missing test — scored every ranking signal against
> **real** activity on our 46 wet-lab MICs **and** 946 independent DBAASP peptides (harvested this session).
> APEX ranks real activity across the full range (Spearman +0.33) but **flattens inside the high-activity band our
> top-100 lives in** (+0.06), while composition (Lysine-richness) predicts it there (+0.27) — a range-restriction
> failure. So APEX is kept only as the active-band gate; composition ranks within it. feat-033a top-50 (seed 42):
> R/(R+K) 0.60→**0.33** (Lys>Arg 14%→94%), aromatic 0.20→0.10, **0% predicted-hemolytic (ESMC max 0.038, better
> than feat-031's 0.073)**, novelty 0/50 >0.80 (top-50 median 0.69, top-100 max 0.80), diversity clean. Library
> **byte-identical to feat-031's save the top-100**, so Phase-2 (diversity 91% NN<0.6 / novelty median 0.60, 1.3%
> >0.80 / uniqueness 1.0) is unchanged. **158 tests pass** (feat-033/034 + feat-036 hardening added `tests/test_composition.py`
> + an APEX thread-pinning guard + penalty-disabled-message & `_frac` formula-pinning tests, for previously-untested paths); compliance PASS; **byte-reproducible** (two default runs →
> top `dc37c540` / lib `06e30960`). APEX-*predicted* profile Broad 0.64/GN 0.85/GP 0.26/MDR 0.33 — the GP/MDR drop
> is APEX mis-scoring the Lysine chemotype; on every real-data test composition selection ties-or-beats APEX
> selection on all 4 categories. **Adopted with the participant's explicit go-ahead** (a strategic call on the
> one-shot graded submission). Official validator **PASSED on the byte-repro-hardened commit `1c091e5`** (feat-034;
> originally on feat-033 `84e2b78`, byte-identical — the byte-fix is byte-neutral): fresh clone + `uv sync` +
> generate ×2, all 8 checks, fresh-clone output byte-identical `06e30960`/`dc37c540`; re-run once more on the final
> commit before submitting. `--composition-weight 0` recovers feat-031. Only the user-gated Kaggle submit remains.
> **See the feat-034 note below for the adversarial-audit hardening + byte-repro fix (2026-09-29).**
>
> **Adversarial code-review hardening (2026-09-29, docs/RESEARCH.md "feat-033 adversarial code review").** A
> delegated review's load-bearing question — is the shipped λ=1.5 hemolysis penalty still the pure-composition
> ordering we validated? — was checked against ground truth for the first time (`scratchpad/verify_lambda.py`,
> `decisive_lambda.py`). Result: our band's phemo is **bimodal** (median 0.031, 24% >0.5), so `comp − λ·phemo`
> acts as a de-facto gate — λ∈{0.5,0.75,1.5} give a near-identical top-50 — and the penalty is **necessary**
> (pure comp ships a 22%-ESMC-hemolytic top-50, max P 0.93). On the matched-regime 46 the penalty ties-or-beats
> pure comp on real activity; the DBAASP "cost" is a broad-regime potency–toxicity artifact that would require
> shipping hemolytic peptides. **λ=1.5 validated; no change to the shipped bytes.** Robustness fixes applied and
> tested (all preserving the byte-identical output): no-selectivity-model fallback → APEX ranking not pure-Lys;
> graceful out-of-band gate (APEX-ordered, no magic sentinel); `aromatic_weight` guard; superseded-flag NOTE;
> corrected README table; shared aromatic set + shared band helper.
>
> **Strategic adversarial audit (5-agent workflow) + BYTE-REPRO FIX (2026-09-29).** A devil's-advocate panel
> attacking the strategic decisions caught one genuine defect the 14-finding review missed: `ApexScorer._run_apex`
> (the sequential APEX path, taken when `_resolve_workers→1`: small input, `device=cuda`, or a low-mem/low-core
> grader) omitted the `OMP/MKL/OPENBLAS/NUMEXPR=1` thread-pinning that `_run_pool` sets — so on such a grader APEX
> runs multi-threaded → non-bit-reproducible → validator check #8 (two-run byte match) could fail = **DQ**. Fixed
> via a shared `_apex_env()` helper (both paths) + regression test; **proven byte-neutral** (our DGX takes the
> parallel path; `_apex_env()` == the old inline env; `dc37c540`/`06e30960` preserved). 156 tests pass. The two
> "worth-testing" challenges were run on cached ground truth (`scratchpad/confirm_audit.py`) and both **confirm
> the status quo**: the λ penalty helps every category on the matched 46 (Broad 0.47/GP 0.36/MDR 0.46 vs band
> 0.27/0.21/0.25), and a muH charge-patterning blend fails the pre-registered bar (re-admits 51.6% hemolytic on
> DBAASP, hurts GP on the 46). Softened a Gram+ overclaim in RESEARCH.md for honesty. **Official validator
> PASSED on the byte-fix commit `1c091e5`** (fresh clone + generate ×2, all 8 checks, fresh-clone output
> byte-identical `06e30960`/`dc37c540` = the approved artifact, confirming the fix is byte-neutral). Also ran
> the audit's Phase-2 *relative* diagnostic (`4108f3b`): corrected a stale FBD (1.938→3.44, pre-mixed-temp) and
> showed our library is 2.7× more novel (near-exact) than the un-ReST base — a defensible Phase-2 entry, no
> library change. HEAD `4108f3b` (docs atop `1c091e5`). Re-run the validator on the final commit before submit.
>
> **feat-031 (now the FALLBACK, `--composition-weight 0`) — the previous shipped default.**
> `--top-temperature 0.8` + `--lys-hedge 0.4` on the same generator. Top-50 (seed 42): **Broad 0.71, Gram- 0.68,
> Gram+ 0.76, MDR 0.67, 0% predicted-hemolytic, µH 0.51** (APEX-predicted); Phase-2 diversity 0.837 / novelty 1.0 /
> uniqueness 1.0. Official validator PASSED on `2f7bb3c`, `c33126c`, and `f4eed63` (fresh clone + generate ×2, all 8
> checks; library `9a3278c9` / top `61becbab`, byte-identical). Superseded because APEX-*ordering* is flat-to-harmful
> inside the selection band (feat-033); feat-031's `--lys-hedge` only nudged an APEX-dominated ranking.
> **feat-032 (Lys-conditioned Gram- ReST generator) — EXPLORED then REVERTED.** A ReST fine-tune (wet-lab Lys
> prior in the reward, gn_w 1.5) produced a generator that *appeared* to strictly dominate feat-031 (APEX Broad
> 0.751, GN 0.749; held-out submodel GN 0.633→0.843) and passed the official validator + seed-robustness. A
> 4-agent adversarial review + verification DISPROVED the win: (1) APEX has ~0 real-activity correlation, so the
> APEX gains don't transfer; (2) the 8 "submodels" are hyperparameter variants of ONE model (train/test r≈0.82),
> so the held-out "generalisation" is mechanical — a **matched APEX-only ReST (lys_w=0) reached the same held-out
> GN 0.829 WITHOUT the wet-lab prior**, confirming ensemble-Goodhart; the GN gain is also easy-species inflation
> (hard K. pneumoniae clears 0.022); (3) it **regressed near-exact novelty ~2.6×** (1.56%→4.08% within 0.80 of a
> known AMP; regenerated Penetratin exactly) — a Phase-2 advancement-axis risk. Its only real edge (composition
> R/(R+K) 0.60→0.50, selectivity 0.0030→0.0012) is thin and feat-031 already banks it via the safe selection
> route. Reverted (checkpoint restored from `experiments/rest/feat031_generator_backup/`); feat-032 checkpoint
> kept at `experiments/rest/lys_gramneg/best.pt`. Lesson: the held-out-submodel guard cannot detect
> ensemble-wide Goodhart — always run the matched control + an independent axis (see docs/RESEARCH.md feat-032).
> **feat-031 (wet-lab Arg-excess hedge):** the decisive non-circular test — scoring the 46 wet-lab peptides
> through APEX — showed APEX has ~0 correlation with REAL success (Gram- rho −0.13, Broad −0.29) and is
> significantly Arg-biased on Gram- (residual vs R/(R+K) +0.41, p=0.003; bootstrap+LOO robust), while reality
> favours Lys-richness (−0.40, p=0.004) — and 75% of the real panel is Gram-. feat-030's Lys-BONUS failed
> (circular APEX metric + wrong form); feat-031's Arg-EXCESS penalty `max(0,R/(R+K)−0.4)` tie-breaks toward the
> abundant APEX-equal BLEND peptides, so it de-biases at ~zero cost. Result: a strict Pareto improvement (APEX
> Broad/GN ↑, GP/MDR held, GP submodel-min ↑), 0%-hemolytic + novelty + amphipathicity preserved, composition
> de-biased 0.82→0.70 (median R/(R+K) 0.667→0.60), **seed-robust (≥ baseline on all 4 cats at seeds 42/43/44)**,
> Phase-2 unchanged, 140 tests pass. PubMed-verified literature corroborates (Hackney 2026 Lys>Arg; Zou 2007;
> Wang 2025) with the "mild tie-breaker, penalize excess not pure-Lys, don't hurt Gram+" refinements adopted.
> **Official validator: PASSED on the shipped feat-031 commit `2f7bb3c`** — fresh GitHub clone + `uv sync` +
> generate ×2, all 8 checks incl. byte-identical reproducibility + ≤80% novelty gate + real ESM++/ESMC path
> ("All checks passed. Submission is valid!"); output library `9a3278c9` / top `61becbab`. Re-confirmed on the
> feat-032-revert commit **`c33126c`**: fresh clone + generate ×2, all 8 checks PASSED, byte-identical output
> `9a3278c9` / `61becbab` — the restored feat-031 checkpoint reproduces the exact validated submission.
> Only **feat-010 (one-shot Kaggle submit)** remains — user-gated (team name + `j_v_v_07` go-ahead; rotate the
> `KGAT_` token).

**Last Updated:** 2026-09-30 (feat-037 — composition-envelope cap SHIPPED as the `uv run generate` default `--max-cationic-fraction 0.444`, SUPERSEDING feat-033 while leaving its core [rank by lysine_fraction within the APEX band] UNCHANGED and the 50k library body byte-identical; demotes the ~20 beyond-envelope (fcat>0.444) extrapolations out of the top-100 → capped top-50 expected Gram+/MDR SR +0.029/+0.026 [honest kNN arbiter on the 46, 95% bootstrap CI excludes 0; HONEST CAVEAT: kNN-model estimate on n=46, removed peptides are unmeasured extrapolations, magnitude ~+0.02–0.03 SR, not a measured wet-lab improvement]. NEW hashes top 21fd02b7 / library bba245dc (pre-cap feat-033 dc37c540/06e30960, recoverable via --max-cationic-fraction 1.0). Official validator re-run PENDING on the final feat-037 commit (top.fasta changed → new hashes). Both feat-035 user flags (top-50-vs-top-100 draw; institutional-email) remain SET ASIDE per the participant. Prior: feat-035 24h deep-validation confirmed feat-033 near-optimal; feat-036 byte-neutral review-response + cross-checks. Commit <feat-037 commit>.)
**Active Feature:** feat-010 SUBMIT (user-gated). **The current shipped/validated state is feat-033 + feat-034, deep-validated by feat-035 (2026-09-30) — see the Current State block at the top of this file; the feat-015…feat-025 narrative that follows is HISTORICAL (feat-021-era), kept for its lessons.** feat-015…feat-020 + **feat-021** (explicit hard
Gram- term, top-50 Gram- 0.54→0.58 at zero Gram+/MDR cost) DONE and **official-validator PASS on the
pushed commit 15c3b5e**. **feat-022** (submodel-cross-validated directed-evolution GA) and **feat-023**
(maximin top-list selection, `--select maximin`) were both explored extensively on the free GPUs and
**rejected as the default** — directed evolution overfits APEX (held-out submodels prove no real gain);
maximin raises the Gram- floor but trades away feat-021's Gram+/MDR/Selectivity standouts, which is a
bad deal when the five categories are ranked separately. **feat-024** (diverse free-GPU exploration:
all-rounder ReST + species/consensus/HC50 checks) — the all-rounder ReST degraded the generator, and
three independent orthogonal checks (multi-predictor consensus, HC50 regressor, ESMFold2 structure) all
CONFIRM feat-021 is sound; documented one real caveat (feat-021's Gram- is E.coli/A.baumannii-only, so
it clears ~0 K.pneumoniae/P.aeruginosa — a biological+oracle limit with no clean fix). **feat-025 —
ADOPTED as the new shipped default:** a closed-form, deterministic **amphipathicity bonus**
(`--amphipathicity-bonus`, smooth Eisenberg-muH floor-ramp `clip((muH-0.25)/0.25,0,1)`, reuses
`physchem`, default 0.0→**0.2**), a mechanistic hedge against APEX's transfer error (the ESMFold2 check
found the bonus-0 top-50 at the weak edge of amphipathicity, 26% non-amphipathic). Measured on the real
400k pipeline: top-50 muH median **0.31→0.40**, non-amphipathic **26%→12%**, at ≤0.014 cost to every
category (Gram+ −0.005, MDR 0.000 — all within APEX's 0.62-AUROC noise) and 0% predicted-hemolytic held;
a pool sweep confirms the hemolysis gate holds at 0/50 across coef 0.0–0.4 and coef 0.2 is the knee. The
"cost" is oracle-internal while the muH gain is oracle-independent, so transfer-adjusted it is net-
positive. Adopted as the default because the organizers run the default entry point; `--amphipathicity-
bonus 0.0` recovers the exact feat-021 selection. **feat-026 — Phase-2 gate measured with the organizers'
own `seqme` framework** (isolated env, kept out of the submission): library Uniqueness 1.0, Diversity
0.839, Novelty 1.0, FBD firmly AMP-like (1.94 vs real-AMP 0.074 / random 5.42) — the advancement gate is
strong. **feat-027 — large-pool (4M, all-8-GPU) anti-Goodhart exploration, REJECTED:** a 10× pool
appeared to lift Gram- 0.569→0.686 / Broad 0.633→0.709 at 0% hemolytic, but the held-out-submodel test
proved it is mostly APEX overfitting (held-out GP collapses 0.75→0.37–0.61, MDR 0.667→0.39–0.57, Gram-
only partially survives); exactly 1 genuine cross-validated all-rounder exists in 4M (K. pneumoniae is
the binding ceiling), too few to shift the categories. The shipped feat-025 top-50 is self-consistent
across submodel splits (robust) and stands — 10× compute confirms it rather than beating it. **feat-028
— mixed-temperature sampling, a GENUINE cross-validated breakthrough, ADOPTED:** feat-027 showed
*selection* is Goodhart-limited, so we optimised the *generator distribution*. A temperature sweep
disproved the "hot sampling has no top-50 cost" claim — cooler sampling lifts the top-50 Gram-/Broad
(the generator's high-activity modes), and it SURVIVES held-out submodel cross-validation (unlike
feat-027). But a cool pool costs Phase-2 diversity. **Mixed-temperature** gets both: the ranked top-100
pool is drawn cool (`--top-temperature 1.0`) while the 50k library body stays hot (`--temperature 1.6`).
Measured on the full pipeline: **top-50 Gram- 0.569→0.640, Broad 0.633→0.680** (GP 0.750, MDR 0.667, 0%
hemolytic, µH 0.40→0.54) with **library diversity 0.837≈0.839 / novelty 1.0 preserved**. All 8 APEX
submodels confirm the mixed top-50's Gram- (mean 0.63 vs shipped 0.56, spread 0.08 — genuine, not
Goodhart). Byte-determinism verified (GPU0==GPU1==rerun). Adopted as the default. 139 tests pass; two
`/code-review` passes addressed. **Re-validation of the new 0.2 default
COMPLETE**: byte-determinism verified across GPUs (runs on GPU0==GPU4==explicit-0.2, identical
`library.fasta`/`top.fasta`), and the **official validator PASSED on commit `6f1d73f`** — fresh clone +
`uv sync` + generate ×2, *"All checks passed. Submission is valid!"*, ranking line
`apex-balanced-success - 1.5*hemolysis + 0.2*amphipathicity`, all 8 checks incl. reproducibility; the
fresh-clone output is byte-identical to the shipped feat-025 default. **Pushed to the public repo
(`d3dd6e6..7995342`), and feat-008 PASSED against the pushed GitHub URL** — fresh GitHub clone + `uv sync`
+ generate ×2, *"Submission is valid!"*, output byte-identical to the shipped default (library `49451d15`,
top `9bb8fe3b`), so the organizers' clone-and-run reproduces exactly this submission. Only feat-010 (the
one-shot Kaggle submit) remains. Pending only the participant's team name +
go-ahead from `j_v_v_07`.
**Deadline:** 2026-09-30 22:00 UTC (1 October 2026, AOE) — see `docs/COMPETITION.md`
**Advancement (phase-2) readiness verified with the organizers' own `seqme` framework (feat-026):**
scored the 50k library exactly as the top-20 screen will — **Uniqueness 1.000, Diversity 0.839
(pairwise Levenshtein), Novelty 1.000** (no exact match to 39,448 known AMPs), 3-gram Jaccard 0.0019;
physchem AMP-like and cationic (charge 4.7, pI 11.6, Gravy −0.54, hydrophobic-moment 0.40); and **FBD
1.94 vs a real-AMP floor 0.074 / random ceiling 5.42** — i.e. "novel AND AMP-like", 3× closer to real
AMPs than to random. Strong on the diversity/novelty/physchem axes the advancement screen scores.

`uv run generate` (feat-021 snapshot — **superseded by the Current State block at the top of this file**; kept for history) produced a scientifically meaningful, reproducible submission: the trained
AR-Transformer samples a novel, cationic/amphipathic 50k library, and the top-100 is ranked by
a **hard Gram+/MDR Success-Rate APEX score** (from an 8×/400k oversampled pool) **minus a λ=1.5
ESMC-600M selectivity penalty**, with a within-list diversity cap, on a generator **ReST-fine-tuned
toward the hard Gram+/MDR categories** (feat-020, selectivity-gated). Byte-reproducible and
structurally cross-checked (ESMFold2). A hard **Gram−** term (`gn_weight=0.75`, feat-021) in the
balanced objective lifts the weakest category at zero Gram+/MDR cost. Final **top-50** (the assayed
set): hard Success-Rate (MIC ≤16 µM) **Broad 0.64, Gram− 0.58, Gram+ 0.75, MDR 0.67**, and **0%
ESMC-predicted-hemolytic** (median P 0.004, max 0.06); novel (median identity 0.62, max 0.73, 0 exact),
Gram+/MDR activity agreed by 85–88% of APEX's 8 sub-models (Gram− 96%). The scientific pipeline
(activity + selectivity + diversity + structure) is complete; remaining is the final
validate-and-submit (user-gated).

## Status

### What's Done

- [x] feat-001 — uv project bootstrap. Python pinned to 3.11, `uv.lock` committed, MIT licensed,
      `generate` wired as the console entry point.
- [x] feat-002 — Compliance layer in `src/amp_challenge_2027/constraints.py`, mirroring the
      official `scripts/verify_submission.py`. 69 tests, covering both the sequence rules and the
      repository-level rules (licence, entry point, pinned environment, defaulted arguments).
- [x] feat-004 — Kaggle entry **verified** via the API as `vijayavallabhj` / `j_v_v_07`:
      `userHasEntered=True`, 0 submissions so far.
- [x] Confirmed the competition ships **no dataset** — its only data file is a 55-byte note reading
      "This is a Hackathon with no provided dataset." All training data is externally sourced and
      must be disclosed, and submission is a write-up rather than a scored file.
- [x] Rule-by-rule audit written up in `docs/COMPLIANCE.md` — every published rule from the Kaggle
      competition rules, Kaggle's Foundational Rules, and the competition website, mapped to where
      it is enforced and how it is verified.
- [x] feat-003 — `uv run generate` writes a contract-valid 50,000-sequence library and ranked
      top 100 in about 2 seconds, byte-identical across processes.
- [x] feat-005 — Training corpus assembled and disclosed. `data/antibacterial.fasta` (39,448 AMPs,
      BSD-3, the organizers' own aggregation of dbAMP/DRAMP/DBAASP/CAMP/APD/+8) loaded via
      `src/amp_challenge_2027/data.py` with full metadata and a deterministic, valid, deduplicated
      `training_sequences()` accessor. Data card in `docs/DATA.md`; disclosed in `SUBMISSION.md`.
- [x] feat-006 — Generator ensemble trained. 8 AR-Transformers (10.68M params) one-per-H100 in
      ~135s. **Finding:** overfits fast on 37k seqs — val loss best at epoch 20 (~1.85), novelty
      falls from 0.95 (ep10) to 0.50 (ep80) as it memorizes. Early-stopping by val loss saves the
      right checkpoint: `best.pt` has novelty ~0.87, samples valid novel cationic/amphipathic
      peptides on CPU. Weights in `training/runs/` (gitignored).
- [x] feat-011 — Trained generator WIRED IN. `checkpoint/generator.pt` (gen6, 10.68M) ships; the
      submission now generates from the trained model, not the placeholder. Deterministic on
      GPU-if-available else CPU; ranks by model likelihood (interim until APEX). Full gate green:
      50k byte-identical across two GPU runs, 101 tests. Design driven by `docs/RESEARCH.md`
      (organizers' baseline = AMP-Diffusion + APEX; our model is CPU-capable, theirs isn't).
- [x] feat-008 — the organizers' own validator passes against the pushed public repo:
      *"All checks passed. Submission is valid!"* Re-run it after any change to the generator.

### What's In Progress

- [ ] feat-006 — Generative model behind `build_model()`
  - Train on `data.training_sequences()`. **Must generalize** beyond the corpus: it is also the
    novelty screen reference, so a memorizing model fails (see `docs/DATA.md`, "dual role").
  - Load weights from `checkpoint/` via `resolve_repo_path`; seed everything from `--seed`.
  - Two official starter kits target this exact format: `szczurek-lab/ampdiffusion-starter-kit`
    and `szczurek-lab/hydramp-starter-kit`.

### What's Next

1. feat-006 — put a real generative model behind `build_model()`.
2. feat-007 — implement a real `score()`; this is what the competition categories measure.
3. feat-009 — fill in the remaining `SUBMISSION.md` sections (abstract, model, ranking).
4. feat-008 — re-run the official verifier, then feat-010 — submit once (from `j_v_v_07`).

## Blockers / Risks

- [ ] **Two Kaggle accounts are reachable from this machine.** The competing account is
      `vijayavallabhj` / `j_v_v_07` (entry verified, `userHasEntered=True`). The *default* token at
      `~/.kaggle/kaggle.json` is a different account (`prakashchhipa`) and reports
      `userHasEntered=False` with a 403 on data. Run `kaggle config view` and confirm the username
      before trusting any API answer or submitting anything — Kaggle allows one account per
      participant. See `docs/COMPLIANCE.md` § Account identity.
- [ ] **Three days to the deadline.** The committed baseline is valid but scientifically empty.
      Treat it as a floor that guarantees a submittable entry, not as a candidate entry.
- [ ] **One entry per model.** There is no resubmission to fix a mistake, so feat-008 (the
      organizers' own validator, run against the pushed public URL) is mandatory before feat-010.
- [ ] **Top-100 novelty screen is the likely failure mode for a real model.** A model trained on
      known AMPs tends to reproduce them; anything above 80% Levenshtein identity to
      `data/antibacterial.fasta` is silently dropped and replaced by the next candidate. If a
      large fraction is rejected, the ranking is no longer the model's ranking.

## Decisions Made

- **uv only, no fallback path.** The organizers verify with `uv sync`; matching their toolchain
  exactly removes a class of environment failure. `uv.lock` and `.python-version` are committed.
- **Compliance encoded as a library module, not a script.** `constraints.py` is imported by the
  generator, so an invalid library fails at generation time with a non-zero exit rather than at
  submission time.
- **Reference set vendored into `data/`.** 4.3 MB, 39,448 sequences, from the official template.
  The generator needs it at run time to exclude exact matches and screen the top 100, and it
  makes the repo self-verifying.
- **`build_model()` as the single swap point.** Filtering, ranking, and writing stay fixed, so
  replacing the model cannot accidentally break the submission contract.
- **Baseline is labelled a placeholder in code and docs.** Uniform random peptides have no
  expected activity; the docstrings say so, to prevent a false baseline being read as a result.

## Files Modified This Session

- `src/amp_challenge_2027/constraints.py` — the competition's hard rules as code
- `src/amp_challenge_2027/data.py` — training corpus loading, metadata, disclosure (feat-005)
- `src/amp_challenge_2027/paths.py` — shared repo-root-aware path resolution
- `docs/DATA.md` — training data card
- `tests/test_data.py` — 21 tests over the data module
- `src/amp_challenge_2027/fasta.py` — FASTA I/O matching the official parser
- `src/amp_challenge_2027/model.py` — `PeptideGenerator` protocol and `RandomBaseline`
- `src/amp_challenge_2027/generate.py` — `uv run generate` entry point
- `tests/test_constraints.py` — 45 tests over the compliance layer
- `scripts/verify_submission.py` — vendored official validator (BSD-3, upstream attribution)
- `data/antibacterial.fasta` — vendored reference set
- `AGENTS.md`, `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh` — harness
- `README.md`, `docs/COMPETITION.md`, `SUBMISSION.md`, `LICENSE`

## Evidence of Completion

- [x] Tests pass: `uv run pytest -q` → `90 passed`
- [x] Generation: `uv run generate` → 50,000 records, 50,000 unique, lengths 8–50,
      0 alphabet violations, top.fasta 100 records all present in the library
- [x] Reproducibility: three fresh processes → library `6e24c32d5ff6b3988b7a1ef9401f5ab4`,
      top `ad0ac45cc621e865636e8bb70b607f24`
- [x] Official validator against the pushed public URL: `uv run python
      scripts/verify_submission.py https://github.com/Vijayavallabh/amp-challenge-2027`
      → "All checks passed. Submission is valid!" (7s, all 8 checks)

## Notes for Next Session

Start with `./init.sh`; it reproduces all of the evidence above in about ten seconds.

The interesting work is feat-006 and feat-007. Note that the two are scored differently: the
50,000-sequence library is screened computationally for diversity, novelty and physicochemical
distributions, while only 25 peptides drawn at random from the **top 50** are synthesized and
assayed. Optimising the library and optimising the top of the ranked list are not the same
problem, and the random draw means a single good sequence cannot carry the entry.

Two official starter kits target this exact submission format and are worth reading before
writing a model from scratch: `szczurek-lab/ampdiffusion-starter-kit` and
`szczurek-lab/hydramp-starter-kit`.

---

## Session 3 (2026-09-28) — optimize-to-deadline (goal: maximize win odds on 8xH100)

**Compute unlocked:** stood up a CUDA torch 2.5.1 env (scratch, gitignored) so APEX scores on
all 8 H100s: **50k peptides in 9.5s (~5.3k/s)**. Shipped path stays CPU-deterministic; GPU APEX is
offline only. Reusable `experiments/bulk_apex.py` (shards over CPU cores or 8 GPUs -> cached npz).

**Objective bake-off (experiments/analyze_objective.py, cached 50k lib):** tested a category-aligned
success-rate objective vs the shipped `broad_potency`. **Hypothesis DISPROVEN / not adopted:** on this
generator's pool, `broad_potency` already gives higher predicted Gram-neg SR (74 vs 64) and Broad SR
(57 vs 55); the category objective only trades GN -> GP/MDR. Keep `broad_potency`-style activity ranking.

**Real weaknesses found (predicted; no wet-lab claims):**
1. Pool is **Gram-negative-leaning**: top-50 predicted SR GN=74%, but **GP=28%**, MDR~35%. Gram-Positive
   coverage is the weak scored category. This is a *generator distribution* gap, not a selection gap.
2. Most APEX-active peptides are predicted **hemolytic** (P~0.76 pre-penalty) -> activity/selectivity tension.

**Plan (prioritized):**
- [feat-015] **ReST generator fine-tuning** on the 8 H100s: sample -> APEX+hemolysis score -> keep
  high-reward NOVEL, diverse samples -> low-LR fine-tune -> repeat. Reward = broad soft-Success-Rate
  (hard Gram+/MDR strains up-weighted) - lambda*P(hemolytic). Guards: novelty+diversity monitored every
  round (Phase-2 gate), physchem-envelope check, held-out APEX sub-model validation, saturating success
  (no chasing APEX's sub-uM tail). Goal: lift GP + selectivity without losing GN/broad. Fan out parallel
  chains (different lambda / GP-weight) across GPUs, pick best by held-out reward.
- [feat-016] **Deterministic parallel CPU-APEX** in shipped `oracle.py` (shard per-seq over cores;
  reassemble by sequence -> byte-identical). Cuts shipped runtime ~20x, affords larger oversample.
- Preserve the validated working submission at all times; re-run official validator before any ship.

**feat-015 ReST — validated + sweep running (2026-09-28):** 1-round smoke test: generator's
whole-pool active fraction **6.5% -> 50.5%**, novelty held 0.916 -> 0.934, charge 2.5 -> 6.0
(real-AMP-like), top-100 SR_broad 28.8 -> 38.5, GN 31.9 -> 46.3, P(hemolytic) 0.137 -> 0.071. No
collapse. Launched an 8-chain parallel Pareto sweep (one per H100) over lambda in {0.5,1,1.5,2} x
gp/mdr up-weight in {0,0.5,1}, 5 rounds each, plus a seed replicate. Diagnostics per round:
novelty (verbatim), div150 (diverse survivors of top-600 at 0.6 cap), physchem envelope, per-bucket
SR, P(hemolytic). Tools in experiments/ (bulk_apex, reward, sample_pool, rest_finetune,
preview_select, apex_permodel, pick_winner). Winner gets a held-out APEX sub-model check + a
reference-novelty check before it can replace checkpoint/generator.pt; then re-run the official
validator. Shipped deterministic path UNCHANGED until then.

**Sweep results + validation (2026-09-28, session 3):**
- 8-chain sweep: **gp_w drives the Gram balance** — gp=0.5 -> GN-specialist (GN~92, GP~26); gp=1.0 ->
  balanced (c3: GN 60/GP 53/MDR 50, phemo 0.046; c5: GN 67/GP 69/MDR 60, phemo 0.17). c1==c7 (seed
  replicate) -> ReST is reproducible. All chains lifted whole-pool active 6.5% -> ~79% over 5 rounds.
- **Anti-Goodhart checks PASS:** (a) top peptides are realistic amphipathic alpha-helical AMPs
  (e.g. ILGKLLSTAAKLLSKL, q=+3..+4, L15-18, breadth 9-10/11, minMIC ~1.5uM), not adversarial noise;
  (b) held-out APEX sub-model agreement 0.95 (>=6/8 submodels agree on 95% of active calls); breadth
  via submodels {0-3}=7.5 vs {4-7}=7.8, corr 0.67 -> activity generalizes across the ensemble, not
  gaming the mean; (c) physchem envelope sane; (d) verbatim novelty preserved (0.92 -> 0.97).
- **Only weakness: diversity collapse** — top-reward region concentrates to ~16 motif families
  (div150 150 -> ~16). Competition rewards a DIVERSE top-50 (25 drawn at random) + a diverse 50k
  library (Phase-2 screen), so this must be fixed. Launched a diversity-aware sweep (--dedup-cap 3,
  MinHash near-dup capping in the fine-tune set) from the diverse base generator: d0 (lam1,gp1),
  d1 (lam0.5,gp1), d2 (lam1,gp0.5), d3 (lam1.5,gp1).
- **feat-016 DONE + validated:** shipped CPU-APEX now shards over single-threaded workers ->
  byte-reproducible AND worker-count-independent (workers8==workers24, max|diff| 0.0), ~2x+ faster,
  memory-guarded auto worker count. 127 tests pass. Enables a larger shipped oversample.

**BREAKTHROUGH — sampling temperature fixes diversity AND novelty (2026-09-28):**
dedup-cap failed to stop collapse, and earlier rounds trade activity for diversity. But sampling
the ReST generator at **temperature 1.3** broadens the active manifold: on c3/round3, library
diversity 30% -> **57%**, novelty median 0.73 -> **0.60** (frac>0.8 0.27 -> 0.095), with top-50
**activity unchanged** (breadth 6.1, GN 71). So the shipped generator can be a strong late-ish ReST
round sampled hot -- active AND diverse AND novel. Decision converging on **c3/round3 + temperature
1.3**, category selection (gp_w 0.5, lambda 0.5, diversity 0.6). Shipped ranker + oracle already
updated (category_success_score); feat-016 parallel APEX makes the larger oversample affordable.
