# AMP Challenge 2027 — the rules we are scored against

Reference notes compiled 2026-09-27 from the official sources. Where this file and the official
sources disagree, the official sources win — re-check before submitting.

For the audit — each rule mapped to where it is enforced and how it is verified — see
[COMPLIANCE.md](COMPLIANCE.md).

- Competition site: https://szczurek-lab.github.io/amp-challenge-website/
- Kaggle entry point: https://www.kaggle.com/competitions/amp-challenge
- Official template and validator: https://github.com/szczurek-lab/amp-challenge-2027
- Questions: https://github.com/szczurek-lab/amp-challenge-2027/issues

The challenge asks for generative models that design novel antimicrobial peptides active against
clinically relevant bacteria, including multi-drug-resistant ESKAPE pathogens. It runs on the
NeurIPS 2026 Competition Track, and the top candidates are synthesized and assayed in a wet lab.

## Timeline

| Date | Milestone |
|---|---|
| April 2026 | Competition call opened |
| **2026-09-30 22:00 UTC** (1 Oct 2026 AOE) | **Submission deadline** |
| October 2026 | Phase 1 computational evaluation |
| 6–12 December 2026 | AMP Challenge at the NeurIPS Competition Track |
| Oct 2026 – Feb 2027 | Peptide synthesis window |
| Mar – Apr 2027 | Experimental validation (MIC / HC50) and final analysis |
| Late 2027 | Benchmarking paper submission |

## Deliverables

**Minimum — benchmark participation.** Teams meeting only these get their experimental results
but are not eligible for co-authorship.

- An abstract summarizing the method
- A library of 50,000 designed peptides
- A ranked list of the top 100, with the selection and ranking procedure documented
- A short summary of training data, external databases, and any manual or computational filters
- A GitHub repository (private is acceptable) with model weights and inference code, granting
  read access to [@RasmusML](https://github.com/RasmusML) and
  [@szymczakpau](https://github.com/szymczakpau)

**Full — co-authorship eligibility.** Everything above, plus:

- A **public** repository following the official template
- A permissive OSI-approved licence (MIT, BSD-3-Clause, or Apache 2.0) — this repo uses MIT
- `uv` for dependency management, with `uv.lock` and a pinned Python version committed
- `uv run generate` as the entry point, producing the library and top list, every argument
  defaulted
- A fixed default random seed, so two runs produce identical output
- Full training data disclosure; any non-public data must itself be released under a permissive
  licence as part of the submission

## The submission contract

The validator (`scripts/verify_submission.py`, vendored here) clones the repo, runs `uv sync`,
runs `uv run generate`, checks the output, then runs it again and compares bytes. It expects:

```
generate/
  library.fasta   exactly 50,000 records
  top.fasta       exactly 100 records
```

Every sequence must:

- use only the 20 standard proteinogenic amino acids, `ACDEFGHIKLMNPQRSTVWY`
- be 8 to 50 residues long
- be unique within its file
- carry a non-empty FASTA header
- be linear with free termini — no amidation or other terminal modification, no non-canonical
  residues, staples, peptidomimetics, lipidation, glycosylation, PEGylation, or dendrimers

Additionally:

- the **library** must contain no sequence identical to any in `data/antibacterial.fasta`
  (39,448 known antibacterial peptides)
- every sequence in the **top list** must also appear in the library, and must not exceed
  **0.80 Levenshtein ratio** against any sequence in that reference set
- both files must be byte-identical across two consecutive runs

All of this is enforced at generation time by `src/amp_challenge_2027/constraints.py`.

## How entries are evaluated

1. **Compliance check** — automated screening of sequence validity, alphabet, length, duplicates,
   metadata completeness, training-data sources, and repository licence. Teams that fail are
   notified and given a fixed window to fix issues before disqualification.
2. **Computational screening** — the full 50,000-sequence library is evaluated with the `seqme`
   framework for diversity, novelty against known AMP databases, and physicochemical property
   distributions, plus exact and near-exact matching against DBAASP, dbAMP and APD. If more teams
   qualify than there is experimental capacity for, this phase ranks them and the **top 20 advance**.
3. **Experimental validation** — from each advancing team's top-100 list, **25 peptides are drawn
   at random from the top 50** and synthesized, giving a cohort of 500 peptides. Candidates above
   80% identity to the reference set are treated as invalid and replaced by the next valid one.
   The random draw is deliberate: it measures whether a model is reliably good, not whether it
   produced one lucky sequence.
4. **Results and publication** — all MIC and HC50 measurements are done in one lab under
   identical protocols, blinded to team identity and in randomized order.

### Categories

Teams are ranked by the arithmetic mean of per-peptide metrics across their evaluation batch.

| Category | Ranked by | Tie-break |
|---|---|---|
| Broad-Spectrum Activity | Overall Success Rate across all 20 strains | MIC90 across all 20 |
| Gram-Positive Activity | Gram-positive Success Rate | MIC50 across Gram-positive |
| Gram-Negative Activity | Gram-negative Success Rate | MIC50 across Gram-negative |
| MDR ESKAPE Activity | MDR Success Rate over 8 MDR isolates | MIC50 across MDR ESKAPE |
| Optimal Selectivity | Safety Window (HC50 / MIC50) | lowest MIC50 across all strains |

Definitions used above:

- **MIC** — minimum inhibitory concentration in µM; highest tested is 64 µM, reported as >64 µM
  when no inhibition is seen.
- **HC50** — half-maximal hemolytic concentration in µM; reported as >128 µM at the assay limit.
- **Potency Threshold** — a peptide counts as active at MIC ≤ 16 µM.
- **Success Rate** — percentage of strains in a panel where the peptide meets that threshold.
- **MIC50 / MIC90** — 50th and 90th percentile MIC across the relevant panel.
- **Safety Window** — HC50 / MIC50 across all 20 strains. For Optimal Selectivity, a peptide must
  meet the Potency Threshold on at least one strain to be counted; inactive sequences are excluded.

### Bacterial panel

20 clinically relevant strains: 15 Gram-negative and 5 Gram-positive, of which 8 are
multi-drug-resistant and scored separately in the MDR category. The Gram-negative set spans
*A. baumannii*, *E. cloacae*, *E. coli*, *K. pneumoniae*, *P. aeruginosa* and *S. enterica*; the
Gram-positive set covers *B. subtilis*, *S. aureus* (including MRSA), and vancomycin-resistant
*E. faecalis* and *E. faecium*. The full strain list with ATCC identifiers is on the competition
site under "Official Bacterial Panel".

## Practical notes

- **One entry per model.** Each group submits one primary entry; meaningfully different methods
  need prior agreement with the organizers. There is no resubmission, so validate first.
- **The top 100 must be a subset of the submitted 50,000-sequence library.**
- **Only generative methods are permitted.**
- **Kaggle's Foundational Competition Rules also apply, and supersede the competition-specific
  rules in any conflict.** The ones that bite here: one account per participant (multiple accounts
  are prohibited), and no private sharing of code or data outside your team — public sharing to all
  participants is fine, which the required public repository already is.
- Accepting the rules creates a direct relationship with the host; Kaggle carries no liability for
  prizes, commitments, or host or participant conduct.
- Participation is free; synthesis and assays are paid for by the organizers.
- No wet-lab expertise is required — all experimental work is done by the de la Fuente lab at the
  University of Pennsylvania under standardized protocols.
- Sequences that fail synthesis or QC are **not** retested, which makes careful selection matter.
- Results are expected within roughly 4–6 months of the deadline. All data is released under
  CC-BY 4.0 after joint publication, and tested peptides are deposited to DBAASP and APD3.
- Kaggle write-ups are licensed CC BY 4.0 and attributed to the Kaggle display name.

## Organizers

Ewa Szczurek's lab (Helmholtz Munich and the University of Warsaw) runs the competition and
purchases synthesis; the Cesar de la Fuente group at the University of Pennsylvania performs the
wet-lab work. Peptides are synthesized through a single vendor workflow (AAPPtec) under common QC.
