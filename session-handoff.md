# Session Handoff

## Current Objective

- **Goal:** Get the improved **feat-037** entry validated and submitted cleanly by the
  **2026-09-30 22:00 UTC** deadline. The design work is complete — the reproducible 50k library +
  top-100 (active, selective, novel, diverse) is shipped and locally byte-reproducible; because the
  feat-037 cap changed `top.fasta`, the **official validator must be re-run on the final commit
  (PENDING)**, after which the only work left is the **user-gated Kaggle submit**.
- **Current status (through feat-037):** **SHIPPED = feat-037 — the composition-envelope cap
  (`uv run generate` default `--max-cationic-fraction 0.444`), which SUPERSEDES feat-033 as the shipped
  default.** feat-033's core (wet-lab-calibrated COMPOSITION ranking — rank by Lysine-richness within the
  APEX active band) is **UNCHANGED and vindicated**; feat-037 only demotes the ~20 beyond-envelope
  (fcat>0.444) extrapolations out of the top-100, so the **50k library body is byte-identical** and only the
  top-100 selection changes. New hashes: **top `21fd02b7aa928f32c1ac6f6aeb1faa2b`, library
  `bba245dccc21be693a80c1b114748bb5`** (pre-cap feat-033 `dc37c540`/`06e30960`, recoverable via
  `--max-cationic-fraction 1.0`). feat-034 (adversarial-audit hardening + byte-reproducibility fix) sits
  unchanged underneath. The prior session, **feat-035 (2026-09-30)**, was a 24-hour deep-validation pass
  with **no artifact change** that confirmed feat-033's *ranking* is near-optimal (see below) — they tested
  re-ranking *within* the validated range, which is within noise. **feat-037 (this session) found the one
  robustly-supported improvement they had not tested: removing the beyond-envelope (fcat>0.444)
  extrapolations, which lifts the two hardest boards (Gram+/MDR).** A same-day follow-up, **feat-036**,
  closed out a background code review and ran two more independent ground-truth cross-checks — all
  confirming ship feat-033, artifact byte-unchanged (see the feat-036 block below). Full feat-001…feat-036
  record lives in `progress.md` + `feature_list.json`; historical feat-016…feat-032 notes there are
  point-in-time and correct as-of-their-date — do not rewrite them.
- **Branch / commit:** `main` @ **`<feat-037 commit>`** (composition-envelope cap, atop the feat-036
  validated code). **feat-037 changed `top.fasta` (new hashes `21fd02b7…`/`bba245dc…`), so the official
  validator MUST be re-run on this final commit — PENDING, not yet re-confirmed for feat-037**; local
  two-run byte-repro on the capped default is being re-confirmed. The **pre-cap feat-033 artifact**
  (`dc37c540`/`06e30960`) had PASSED the official validator on `a7ed06f` (2026-09-30: fresh clone + `uv sync`
  + generate ×2) and earlier on `1c091e5` (feat-034; artifact `84e2b78`). Submission remains **user-gated** —
  team name + explicit go-ahead from **`j_v_v_07`** only, never the machine-default Kaggle token (a different
  account).

## Shipped submission (current truth — feat-037: feat-033 ranking + envelope cap + feat-034)

- `uv run generate` defaults to **`--composition-weight 1.0 --aromatic-weight 0.5 --max-cationic-fraction
  0.444`** on the feat-020/021 generator (`checkpoint/generator.pt` = `da70fb42`, unchanged). It ranks the
  top-100 by **`lys_fraction − 0.5·aromatic_fraction − 1.5·P(hemolytic)` within an APEX-active-band gate**,
  NOT by APEX potency, and then **caps the selection to the wet-lab-validated cationic envelope (feat-037)**
  — demoting any candidate with `(K+R)/len > 0.444` below every in-envelope one, which keeps the assayed
  top-100 inside the validated envelope. Shipped hashes: top **`21fd02b7…`**, library **`bba245dc…`**
  (`--max-cationic-fraction 1.0` recovers the pre-cap feat-033 selection `dc37c540`/`06e30960`).
- **Why composition, not APEX potency (feat-033, docs/RESEARCH.md):** scored against **real** activity
  on 46 wet-lab MICs + 946 independent DBAASP peptides, APEX ranks real activity across the full range
  (Spearman +0.33) but **flattens inside the high-activity band the top-100 lives in** (+0.06), while
  composition (Lysine-richness) still predicts it there (+0.27) — a range-restriction failure. APEX is
  kept only as the active-band **gate**; composition ranks within it. On every real-data test,
  composition selection ties-or-beats APEX selection on all 4 categories.
- **Fallback:** **`--max-cationic-fraction 1.0` recovers the pre-cap feat-033 selection**
  (`dc37c540`/`06e30960`); **`--composition-weight 0` recovers the earlier feat-031 APEX ranking** (previous
  shipped default: `--top-temperature 0.8 --lys-hedge 0.4`, validator-PASSED on `2f7bb3c`/`c33126c`/
  `f4eed63`).
- **feat-034 (2026-09-29):** adversarial-audit hardening + a **byte-reproducibility fix** — the
  sequential APEX path (`ApexScorer._run_apex`) omitted the `OMP/MKL/OPENBLAS/NUMEXPR=1` thread-pinning
  that the pooled path sets, a latent disqualification on a grader that takes that path. Fixed via a
  shared `ApexScorer._apex_env()` helper (both paths) + a regression test; **proven byte-neutral**
  (`dc37c540`/`06e30960` preserved). The λ=1.5 hemolysis penalty was validated against ground truth
  (our band's P(hemo) is bimodal, so the penalty acts as a de-facto gate and is necessary). No change
  to the shipped bytes.

## feat-035 (2026-09-30): 24h deep validation, NO artifact change

A full-day GPU-backed pass (ground truth + a 5-agent literature deep-research + offline GPU
experiments; shipped artifact `dc37c540` untouched) re-validated feat-033 from new angles and found no
shippable improvement. Detail: `docs/RESEARCH.md` "feat-035" + `scratchpad/SESSION_FINDINGS.md`.

- **Optimal Selectivity validated on REAL HC50/MIC** (109 DBAASP peptides with both labels — real, not
  APEX, so not circular): the composition chemotype top-20 has a median real safety window **92.7 vs
  58.5 baseline** at equal potency. Our strongest, least-contested board.
- **APEX active-band gate tested keep-vs-drop → KEEP.** On the 46 novel/OOD peptides the gate *looks*
  harmful, but the bootstrap gap CI **[−0.02, +0.52] includes 0** (not robust, n=46); on DBAASP it
  *looks* helpful but is **circular** (APEX = the competition lab, trained on DBAASP-like data →
  DBAASP is in-distribution). Dropping it would be the feat-032 trap.
- **Learned/TabPFN-style ranker collapses out-of-distribution** on novel peptides, while the
  composition formula transfers (Spearman **+0.448**). No learned ranker adopted.
- **No validated Gram+ or MDR lever exists** (composition +0.12 ns for Gram+; MDR is unvalidatable;
  literature agrees Gram+ is harder). feat-033 correctly optimizes the validatable boards
  (Gram−/Broad/Selectivity) without sacrificing them.
- **Recommendation: ship feat-033 as-is.**

## feat-036 (2026-09-30): code-review closeout + independent cross-checks + byte-neutral hardening

A same-day follow-up to feat-035. A background `/code-review` found no surviving correctness bug; its top
finding + two further independent, ground-truth-anchored experiments all reconfirm feat-033. The shipped
artifact is **byte-unchanged** (a 2× `generate` run on the patched code reproduced `dc37c540`/`06e30960`);
**158 tests** pass (156 + 2). Detail: `docs/RESEARCH.md` "feat-035 code-review closeout" and "independent
selectivity cross-check"; `scratchpad/codereview_finding1.md`, `hemopi2_finding.md`, `grampos_finding.md`.

- **The selectivity λ=1.5 is a GATE, not a miscalibration.** The review flagged λ (calibrated for the
  larger `balanced_success_score` scale) as maybe dominating composition. On the shipped band it acts as a
  gate: band `phemo` is bimodal, λ clears the ~54% hemolytic mode, and the top-100 is 99% clean and
  **composition-ordered** (Spearman(final,comp)=+0.90 vs (final,−phemo)=+0.06; overlap with a
  selectivity-only ranking = 0%). Rescaling λ to the composition scale (the review's suggestion) would
  re-admit hemolytic peptides for ~zero composition gain — **no λ change.**
- **Byte-neutral honesty fixes applied** (review #2/#3/#5): distinguish "penalty disabled" from "model
  unavailable" in the `build_ranker` fallback message; add an explicit `SELECTION:` provenance line under
  `--select maximin`; pin the `lys/aromatic_fraction` formula to `physchem._frac` with a test. All are
  stdout/test-only on paths the shipped default never runs — output byte-identical.
- **Independent selectivity cross-check (HemoPI2 recipe) → corroborates feat-033.** A reconstructed
  HemoPI2-class RF (independent-set R 0.702 vs paper 0.739) rated the top-100 mildly hemolytic, disagreeing
  with our ESMC head — but the **real-HC50 arbiter** resolves it in our favour (our Lys regime is
  in-distribution and genuinely safe, median real HC50 133 µM). The disagreement is generated-peptide OOD:
  a *third* learned model (after APEX, ESMC) failing OOD on novel peptides, while composition-on-real-data
  transfers. Honest caveat recorded: our own "0% hemolytic" is likewise an ESMC OOD estimate — trust the
  real-HC50-validated chemotype, never a per-peptide guarantee.
- **No safe Gram+ lever (uncontested board) — confirmed.** No physchem feature specifically predicts real
  Gram+; the strongest (net_charge) helps Gram− more and *reverses sign* OOD (DBAASP +0.22 vs the 46 novel
  −0.31), and our top-100 is already well-charged (median 7.0) — chasing Gram+ with charge would backfire.

## feat-037 (2026-09-30): composition-envelope cap — SHIPPED, supersedes feat-033 as the generate default

The first shipped-artifact change since feat-033, and a disciplined one: feat-033's core (rank by
Lysine-richness within the APEX band) is unchanged and vindicated; feat-037 adds a final **composition-envelope
cap** (`--max-cationic-fraction 0.444`, the new default) that demotes any top candidate whose cationic fraction
`(K+R)/len` exceeds 0.444 — the max among the 46 wet-lab actives — below every in-envelope one, so the assayed
top-100 stays inside the validated envelope. Library body byte-identical; only the top-100 changes (63/100
differ). Detail: `docs/RESEARCH.md` "feat-037".

- **Why.** The uncapped composition ranker extrapolated — ~20/100 of the top-100 sat past the validated envelope
  (fcat>0.444), concentrated at the head (13 of the top-50; pos 1 = `KKKLKKKLLKKKKKLRLL`, fcat 0.667), ~6×
  enriched vs the library. On the 46 wet-lab MICs that region is where **real Gram+ activity collapses**
  (fcat>0.40 → Gram+ SR ~0.07 vs peak ~0.36). The library is healthy; the Goodhart was in the *ranking*, so the
  fix is a re-ranking.
- **Evidence (honest kNN arbiter on the 46, LOO-validated: broad +0.48/gn +0.43/gp +0.29/mdr +0.46).** Capped
  top-50 expected Success Rate vs feat-033 (paired bootstrap B=5000): **Gram+ +0.029 (95% CI [+0.003,+0.047])**
  and **MDR +0.026 (CI [+0.001,+0.054])** exclude 0; Gram− +0.023 and Broad +0.022 are directional (CI includes
  0). Robust at k=5 and k=7 for the top-50, positive at all k (3/5/7) and both pools, never negative. 0.444 is
  optimal — a 0.40 cap loses the Gram− gain and collapses diversity to 41/100. Mechanism: the 20 removed peptides
  are below-average SR on every board, most on Gram+ (0.177 vs the 0.242 top-50 mean).
- **New hashes:** top **`21fd02b7aa928f32c1ac6f6aeb1faa2b`**, library **`bba245dccc21be693a80c1b114748bb5`**.
  3 new cap tests in `tests/test_composition.py` (demote-over-cationic / in-envelope-unchanged /
  None-recovers-feat033-order); suite green. Recovery: `--max-cationic-fraction 1.0` (pre-cap feat-033),
  `--composition-weight 0` (feat-031 APEX).
- **Honest caveat.** The gain is a **kNN-model estimate on n=46**; the removed peptides are extrapolations with
  **no direct wet-lab measurement** (magnitude modest, ~+0.02–0.03 SR; top-100/k=3 not CI-robust; diversity
  marginally lower). An explicit, ground-truth-supported, one-shot-appropriate refinement, **not a measured
  wet-lab improvement** — the standing "all figures are computational predictions" caveat holds.
- **Validator.** `top.fasta` changed → the official validator MUST be **re-run on the final pushed commit
  (PENDING, not yet claimed passed)**.
- **Also explored, not shipped:** D1 (ESM-C 600M masked-infill ~46k) + ProGen2 (from
  `anthropics/uplifting-biomolecular-modeling`, ~42k in-envelope/novel/diverse) diversification pools —
  floor-play material for a hedged library only, activity unproven, needs a risky Tier-2 library change; D3 — an
  MC-corrected 16-feature battery found no transferable Gram+ physchem lever (every DBAASP signal sign-flips OOD;
  confirms feat-035/036).

## Final submission characterisation (feat-037 capped, local full 50k run, seed 42)

- **Top-50 (capped):** R/(R+K) **0.286** (Lys>Arg in 88%), aromatic fraction **0.103**, `fcat` median
  **0.400 / max 0.438** (was 0.667), lys_fraction 0.278 (unchanged), net charge +7, length 18, muH 0.331,
  **0% predicted-hemolytic** (ESMC median 0.001, max **0.052** — still <0.5, and better than feat-031's
  0.073), novelty **0/50 > 0.80** (top-50 median identity 0.69; top-100 max 0.80, rule ≤0.80), within-list
  diversity clean. APEX-*predicted* profile ≈ Broad 0.64 / GN 0.85 / GP 0.26 / MDR 0.33 (same Lysine
  chemotype; the GP/MDR figures are APEX mis-scoring the Lysine chemotype, not a real regression —
  composition ties-or-beats APEX on all 4 real-data boards).
- **Library:** 50,000 unique/valid/novel; **byte-identical to feat-031's save the top-100**, so Phase-2
  is unchanged — diversity 91% (NN < 0.6), novelty median 0.60 (1.3% > 0.80), uniqueness 1.0.

## Verification evidence

| Check | Command | Result |
|---|---|---|
| Tests | `uv run pytest -q` | **161 passed** (158 + feat-037's 3 cap tests in `tests/test_composition.py`: demote-over-cationic / in-envelope-unchanged / None-recovers-feat033-order; earlier: feat-033/034 composition + APEX thread-pinning guard, feat-036 penalty-disabled-message + `physchem._frac` formula-pinning) |
| Full 50k reproducibility (APEX + ESMC PLM on GPU) | two `uv run generate` runs | feat-037 capped: top `21fd02b7…`, library `bba245dc…` (two-run byte-repro being re-confirmed); pre-cap feat-033 was byte-identical top `dc37c540` / library `06e30960` |
| Official validator | `verify_submission.py <repo-url>` | **Re-run PENDING on the final feat-037 commit** (top.fasta changed → new hashes `21fd02b7…`/`bba245dc…`). Pre-cap feat-033 PASSED all 8 checks on `a7ed06f` (2026-09-30, fresh clone + `uv sync` + generate ×2; byte-identical `dc37c540`/`06e30960`) and earlier on `1c091e5` |
| Compliance | `docs/COMPLIANCE.md` rule-by-rule audit | **PASS** |

## Key decisions / devil's-advocate findings

- **Rank by composition inside the active band, not by APEX potency** (feat-033): APEX loses ranking
  power inside the high-activity band the top-100 is drawn from (range restriction); composition
  (Lys-rich, low-aromatic) is the one signal whose direction holds on the 46 wet-lab MICs *and*
  independent DBAASP.
- **Keep the APEX gate** (feat-035): the "gate-harmful" signal on 46 novel peptides is not statistically
  robust, and the "gate-helpful" DBAASP result is circular.
- **No learned ranker** (feat-035): it collapses OOD on novel peptides; the composition formula
  transfers.
- **Selectivity is the differentiator** and is validated on real HC50/MIC, not just predicted; the modal
  competitor is APEX-first.
- Heavy models (APEX subprocess, ESMC-600M selectivity head, ESMFold2 structural cross-check) stay
  isolated/offline; the shipped runtime is deterministic with graceful fallback, so a valid submission is
  preserved at all times.

## Open items — blockers

> **The two user flags raised earlier this session have been SET ASIDE by the participant (2026-09-30) —
> do NOT treat them as blockers or raise them again.** They are kept below only as a record of the due
> diligence; the participant has the context to judge them and has chosen not to pursue them.

- ~~**FLAG 1 — wet-lab draw ambiguity** (site/design-PDF "top 100" vs FAQ "top 50"; if top-100, positions
  51–100 are also assayed and composition tapers there).~~ *Set aside per the participant.* The design hedge
  (a uniformly strong top-100) already covers both cases regardless.
- ~~**FLAG 2 — institutional-email registration requirement** vs the gmail account.~~ *Set aside per the
  participant.*

**Standing blockers (one-shot submission — get these exactly right):**

- [ ] **One entry, no resubmission** — **feat-037 changed the code (new `--max-cationic-fraction 0.444`
      default) and the artifact (`top.fasta` → new hashes `21fd02b7…`/`bba245dc…`), so the official validator
      re-run is REQUIRED and PENDING on the final feat-037 commit** (this is a code+artifact change, not
      docs-only). The pre-cap feat-033 artifact had PASSED all 8 checks on `a7ed06f` (2026-09-30, fresh-clone
      byte-identical `dc37c540`/`06e30960`); re-run `scripts/verify_submission.py <repo-url>` on the final commit
      before submitting. Submit only from **`j_v_v_07`** (never the machine-default token, which belongs to a
      different account — a one-account-rule breach).
- [ ] **Rotate the exposed `KGAT_` token** before submitting (it goes in `KAGGLE_API_TOKEN`, not
      `kaggle.json`).
- [ ] Oracles are estimates, not measurements (APEX AUROC 0.62–0.76; ESMC selectivity 0.905; composition
      validated on real HC50/MIC but still a design-time screen) — **never claim wet-lab efficacy.**

## Recommended next step

The shipped artifact is now the **feat-037 composition-envelope-capped** selection (top `21fd02b7…`, library
`bba245dc…`); feat-035/036 confirmed feat-033 near-optimal across ~9 independent angles, and feat-037 removes the
one identified extrapolation risk (beyond-envelope peptides) without adding modelling scope. Do **not** add
further modelling scope (headroom is bounded; every apparent lever, incl. learned rankers, net-charge, HemoPI2
and amphipathicity, collapses OOD). The two user flags are **set aside per the participant**. Remaining actions:

1. Rotate the `KGAT_` token.
2. **Re-run `scripts/verify_submission.py <repo-url>` on the final feat-037 commit — REQUIRED and PENDING**
   (feat-037 changed the code + `top.fasta`; the pre-cap `a7ed06f` pass no longer covers the shipped artifact).
   Expect the new hashes `21fd02b7…`/`bba245dc…`.
3. Submit from **`j_v_v_07`** with the participant's team name and explicit go-ahead.

## Startup

1. Read `AGENTS.md`, `docs/COMPETITION.md`, `feature_list.json`, `progress.md` (Current State).
2. `./init.sh` (installs, tests, generates twice, checks byte-reproducibility; needs network for the
   isolated APEX env on first call, and a GPU for a fast full run — falls back to CPU/likelihood
   otherwise).
3. `git log --oneline -5`.
