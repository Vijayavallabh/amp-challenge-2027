# Session Handoff

## Current Objective

- **Goal:** Get the frozen, validated entry submitted cleanly by the **2026-09-30 22:00 UTC**
  deadline. The design work is complete — the reproducible 50k library + top-100 (active, selective,
  novel, diverse) is shipped and validator-PASSED. The only work left is the **user-gated Kaggle
  submit**.
- **Current status (through feat-035):** **SHIPPED = feat-033 (wet-lab-calibrated COMPOSITION
  ranking) + feat-034 (adversarial-audit hardening + a byte-reproducibility fix; the shipped peptides
  are UNCHANGED).** The latest session, **feat-035 (2026-09-30)**, was a 24-hour deep-validation pass
  with **no artifact change** that **confirmed feat-033 is near-optimal** (see below). No safe,
  robustly-supported, shippable improvement over feat-033 exists. Full feat-001…feat-035 record lives
  in `progress.md` + `feature_list.json`; historical feat-016…feat-032 notes there are point-in-time
  and correct as-of-their-date — do not rewrite them.
- **Branch / commit:** `main` @ **`0adffbc`** (docs atop the validated artifact). Official validator
  **PASSED on `1c091e5`** (feat-034; byte-identical to the feat-033 artifact `84e2b78`). Submission
  remains **user-gated** — team name + explicit go-ahead from **`j_v_v_07`** only, never the
  machine-default Kaggle token (a different account).

## Shipped submission (current truth — feat-033 + feat-034)

- `uv run generate` defaults to **`--composition-weight 1.0 --aromatic-weight 0.5`** on the
  feat-020/021 generator (`checkpoint/generator.pt` = `da70fb42`, unchanged). It ranks the top-100 by
  **`lys_fraction − 0.5·aromatic_fraction − 1.5·P(hemolytic)` within an APEX-active-band gate**, NOT
  by APEX potency.
- **Why composition, not APEX potency (feat-033, docs/RESEARCH.md):** scored against **real** activity
  on 46 wet-lab MICs + 946 independent DBAASP peptides, APEX ranks real activity across the full range
  (Spearman +0.33) but **flattens inside the high-activity band the top-100 lives in** (+0.06), while
  composition (Lysine-richness) still predicts it there (+0.27) — a range-restriction failure. APEX is
  kept only as the active-band **gate**; composition ranks within it. On every real-data test,
  composition selection ties-or-beats APEX selection on all 4 categories.
- **Fallback:** **`--composition-weight 0` recovers the earlier feat-031 APEX ranking** (previous
  shipped default: `--top-temperature 0.8 --lys-hedge 0.4`, validator-PASSED on `2f7bb3c`/`c33126c`/
  `f4eed63`).
- **feat-034 (2026-09-29):** adversarial-audit hardening + a **byte-reproducibility fix** — the
  sequential APEX path (`ApexScorer._run_apex`) omitted the `OMP/MKL/OPENBLAS/NUMEXPR=1` thread-pinning
  that the pooled path sets, a latent disqualification on a grader that takes that path. Fixed via a
  shared `ApexScorer._apex_env()` helper (both paths) + a regression test; **proven byte-neutral**
  (`dc37c540`/`06e30960` preserved). The λ=1.5 hemolysis penalty was validated against ground truth
  (our band's P(hemo) is bimodal, so the penalty acts as a de-facto gate and is necessary). No change
  to the shipped bytes.

## Latest session — feat-035 (2026-09-30): 24h deep validation, NO artifact change

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

## Final submission characterisation (feat-033, local full 50k run, seed 42)

- **Top-50:** R/(R+K) **0.33** (Lys>Arg in 94%), aromatic fraction **0.10**, **0% predicted-hemolytic**
  (ESMC max 0.038, better than feat-031's 0.073), novelty **0/50 > 0.80** (top-50 median identity 0.69;
  top-100 max 0.80, rule ≤0.80), within-list diversity clean. APEX-*predicted* profile Broad 0.64 /
  GN 0.85 / GP 0.26 / MDR 0.33 — the GP/MDR drop is APEX mis-scoring the Lysine chemotype, not a real
  regression (composition ties-or-beats APEX on all 4 real-data boards).
- **Library:** 50,000 unique/valid/novel; **byte-identical to feat-031's save the top-100**, so Phase-2
  is unchanged — diversity 91% (NN < 0.6), novelty median 0.60 (1.3% > 0.80), uniqueness 1.0.

## Verification evidence

| Check | Command | Result |
|---|---|---|
| Tests | `uv run pytest -q` | **156 passed** (feat-033/034 added `tests/test_composition.py` + an APEX thread-pinning guard) |
| Full 50k reproducibility (APEX + ESMC PLM on GPU) | two `uv run generate` runs | **byte-identical** (top `dc37c540`, library `06e30960`) |
| Official validator | `verify_submission.py <repo-url>` | **All 8 checks passed** on `1c091e5` (fresh clone + `uv sync` + generate ×2; fresh-clone output byte-identical `06e30960`/`dc37c540`) |
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

## Open items — user flags (confirm with organizers) & blockers

**Top priority — TWO USER FLAGS raised this session, to CONFIRM WITH THE ORGANIZERS before the one-shot
submit** (AGENTS.md escalation: check `docs/COMPETITION.md`, then the official FAQ / a GitHub issue on
`szczurek-lab/amp-challenge-2027`; anything touching what gets submitted goes to the user):

- [ ] **FLAG 1 — wet-lab draw is ambiguous in the organizers' OWN materials.** The website
      "How-It-Works" section and the design PDF say **25 are drawn at random from the TOP 100**; the
      website **FAQ says the TOP 50**. If it is top-100, list positions **51–100 are also assayed** —
      and our composition ranking **tapers there**. Resolve before submitting.
- [ ] **FLAG 2 — registration reportedly requires an INSTITUTIONAL email** ("gmail/hotmail/yahoo not
      accepted", per the competition materials); the entry uses a **gmail account** (`j_v_v_07` /
      `vallabh2006@gmail.com`). Confirm eligibility before submitting (`docs/COMPLIANCE.md` § Account
      identity).

**Standing blockers (one-shot submission — get these exactly right):**

- [ ] **One entry, no resubmission** — re-run `scripts/verify_submission.py <repo-url>` on the **final
      HEAD** immediately before submitting, and submit only from **`j_v_v_07`** (never the machine-default
      token, which belongs to a different account — a one-account-rule breach).
- [ ] **Rotate the exposed `KGAT_` token** before submitting (it goes in `KAGGLE_API_TOKEN`, not
      `kaggle.json`).
- [ ] Oracles are estimates, not measurements (APEX AUROC 0.62–0.76; ESMC selectivity 0.905; composition
      validated on real HC50/MIC but still a design-time screen) — **never claim wet-lab efficacy.**

## Recommended next step

The artifact is frozen and validator-PASSED, and feat-035 confirmed it near-optimal — do **not** add
modelling scope (headroom is bounded and learned rankers collapse OOD). The **only remaining action is
the user-gated Kaggle submit**:

1. Resolve **FLAG 1** and **FLAG 2** with the organizers.
2. Rotate the `KGAT_` token.
3. Re-run `scripts/verify_submission.py <repo-url>` on the final HEAD (all 8 checks green,
   byte-identical `06e30960`/`dc37c540`).
4. Submit from **`j_v_v_07`** with the participant's team name and explicit go-ahead.

## Startup

1. Read `AGENTS.md`, `docs/COMPETITION.md`, `feature_list.json`, `progress.md` (Current State).
2. `./init.sh` (installs, tests, generates twice, checks byte-reproducibility; needs network for the
   isolated APEX env on first call, and a GPU for a fast full run — falls back to CPU/likelihood
   otherwise).
3. `git log --oneline -5`.
