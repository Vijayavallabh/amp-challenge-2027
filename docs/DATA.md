# Training data card

Full disclosure of the training corpus, as the competition's full requirements demand
(*"Full training data disclosure; any non-public data must be released under a permissive
license as part of the submission"*). Every figure below is reproducible with
`uv run python -c "from amp_challenge_2027.data import corpus_stats; print(corpus_stats())"`.

## Source

| | |
|---|---|
| File | `data/antibacterial.fasta` |
| Origin | Vendored verbatim from the official challenge template, [szczurek-lab/amp-challenge-2027](https://github.com/szczurek-lab/amp-challenge-2027) |
| Licence | **BSD-3-Clause** (the template's licence; retained at `data/UPSTREAM_TEMPLATE_LICENSE.txt`) |
| Size | 4,318,298 bytes; MD5 `8366eb2c5760d614b68f000eee57076d` |
| Records | 39,448 peptides |

This is the organizers' own curated aggregation of publicly available antimicrobial
peptides, released by them as part of the challenge. Using it as training data gives the
cleanest possible provenance and licensing: the data is already public, already
permissively licensed, and already the reference the submission is screened against.

## Why this corpus, and the decision not to pull more (for now)

The competition provides **no dataset**. Training data must therefore be sourced
externally and disclosed, and — for co-authorship eligibility — must be publicly
redistributable under a permissive licence. The vendored `antibacterial.fasta` satisfies
all of that on its own:

- It is **already public and BSD-3-licensed** via the official template, so there is no
  redistribution question to resolve.
- It **aggregates exactly the databases the competition endorses** — the website names
  DBAASP, APD and Peptipedia as open-access sources, and this file draws from DBAASP, APD,
  dbAMP, DRAMP and nine others (see below).
- It is **already pre-filtered to the competition's own constraints** (20 standard amino
  acids, length 8–50, unique), so it needs no cleanup that could introduce error.

Pulling additional raw data from DBAASP / APD3 / dbAMP / Peptipedia directly was
considered and deliberately deferred: each has its own terms of use that would need a
per-source redistribution review, and those databases overlap heavily with this
aggregation anyway, so the marginal gain is small against the licensing risk and the
three-day deadline. If the corpus is later enriched, each new source gets its own row in
the *Source* table above and its own terms-of-use note here before it is used.

## Composition

39,448 peptides, all annotated `antimicrobial`. Lengths 8–50 residues (mean 18.7).
**77.9% carry a net-positive charge**, consistent with the cationic character typical of
antimicrobial peptides — a useful signal for the ranking model (feat-007).

### Source databases

Each peptide records which public databases it appears in (the header `dbs=` field).
Counts are the number of the 39,448 corpus peptides found in each:

| Database | Peptides | Share |
|---|---:|---:|
| dbAMP | 20,933 | 53.1% |
| DRAMP | 18,395 | 46.6% |
| DBAASP | 14,496 | 36.7% |
| CAMP | 11,898 | 30.2% |
| SATPdb | 9,740 | 24.7% |
| APD | 2,041 | 5.2% |
| AMPDB | 1,270 | 3.2% |
| DADP | 601 | 1.5% |
| InverPep | 424 | 1.1% |
| CancerPPD | 396 | 1.0% |
| BaAMPs | 182 | 0.5% |
| CyBase | 169 | 0.4% |
| ParaPep | 124 | 0.3% |

Shares sum to more than 100% because a peptide is typically recorded in several databases.

### Activity annotations

| Activity | Peptides | Share |
|---|---:|---:|
| antimicrobial | 39,448 | 100.0% |
| antibacterial | 7,827 | 19.8% |
| anti-gram− | 5,868 | 14.9% |
| anti-gram+ | 1,776 | 4.5% |
| anticancer | 396 | 1.0% |
| anti-biofilm | 185 | 0.5% |
| antiparasitic | 124 | 0.3% |
| antiviral | 9 | 0.0% |

## Preprocessing and filters applied

The vendored file is used as-is for the novelty screen (it must match the organizers'
file byte for byte). For **training**, `data.training_sequences()` applies a defensive
pipeline:

1. **Validity filter** — keep only sequences that pass the competition constraints
   (20 standard residues, length 8–50). On the vendored corpus this drops **nothing**
   (all 39,448 already comply); it exists so a model is never trained on a sequence it
   could not legally emit, and so a future re-vendored corpus stays safe.
2. **De-duplication** — first occurrence kept, via an insertion-ordered dict (never set
   iteration), so training order is identical across processes. The vendored corpus has
   no duplicates, so this is also a no-op today.
3. **Optional length bounds** — callers may narrow to a length window (e.g. a
   fixed-length model), which is the only filter that changes the count.

No manual curation, hand-picking, or removal of individual sequences is applied. No
external predictor or label is used to select training data.

## The dual role of this file — a disclosed, intentional tension

`data/antibacterial.fasta` is **both** the training corpus **and** the novelty screen
reference. The consequences, by design:

- The submitted **library** may contain **no sequence identical** to it.
- The **top 100** must stay at or **below 80% Levenshtein identity** to every sequence in
  it.

So a model that memorizes this corpus fails: reproducing a training peptide places a
forbidden sequence in the library, and near-copies are stripped from the top 100. The
model must **generalize** to genuinely novel sequences. This is the intended difficulty
of the challenge, the screen that enforces it is already built (`constraints.py`,
`generate.py`), and feat-006 must be designed with it in mind (sampling with temperature,
latent-space variation, and so on — not maximum-likelihood reconstruction).

## Reproducibility

```bash
uv run python -c "from amp_challenge_2027.data import corpus_stats; print(corpus_stats())"
uv run pytest tests/test_data.py -q
```

The statistics in this card are asserted in `tests/test_data.py`, so a change to the
corpus that would invalidate them fails CI.

---

## Oracle and validation data (feat-007)

Beyond the training corpus, two external assets are used at **selection/ranking** time — the
part of a submission the competition wet-lab tests. Both are permissively licensed and
redistributable, so co-authorship eligibility is preserved. Neither is used to *train* the
generator; they rank and validate.

| Asset | Path | Origin | Licence | Redistributable |
|---|---|---|---|---|
| APEX-pathogen (8-model MIC ensemble, ~220 MB) | `oracle/apex/` | Wan / de la Fuente lab, *Nat. Microbiol.* 2025 (`machine-biology-group-public/apex-pathogen`); vendored via `szczurek-lab/ampdiffusion-starter-kit` | **MIT** (© 2025 Fangping Wan, `oracle/apex/LICENSE`) | **Yes** |
| Measured MICs (46 peptides × 11 pathogens, 506 rows) | `data/experimental/mic.csv` | Torres et al., *Cell Biomaterials* 2025, doi:10.1016/j.celbio.2025.100183 | **CC BY 4.0** | **Yes**, with attribution |

- **APEX-pathogen** predicts per-species MIC (µM) against the 11 clinical pathogens the
  challenge targets, and is the exact release used to score the AMP-Diffusion baseline library.
  It is vendored as an **isolated `uv` project** (its own `pyproject.toml`/`uv.lock`, pinned
  `torch==2.5.1`+`numpy<2`, Python 3.10) and invoked as a subprocess by
  `src/amp_challenge_2027/oracle.py`, so its dependencies never co-resolve with this package's.
  Weights are committed **directly, not via Git LFS** — the official validator does a plain
  `git clone` with no `git lfs pull`, so LFS pointers would break a fresh checkout.
- **`mic.csv`** is used only to *validate* the oracle (see `docs/RESEARCH.md` § APEX validation),
  never as training data. `mic_relation = ">"` marks a right-censored result (no inhibition up
  to the 64 µM assay ceiling). Attribution: Torres et al. 2025 (CC BY 4.0).

### Hemolysis / selectivity model data (feat-013)

The shipped selectivity model (`checkpoint/hemolysis.pt`, an 11-descriptor MLP) is trained on
**HemoPI-2** — 442+110 high-hemolytic and 370+92 low-hemolytic real peptides from Hemolytik,
with the standard train / held-out-validation split. It is used to rank *degree* of hemolysis
among active peptides (the Optimal Selectivity category), not as generator training data.

| Asset | Path | Origin | Licence | Redistributable |
|---|---|---|---|---|
| HemoPI-1/2 peptide sets | `data/hemolysis/HemoPI*.fasta` | Plisson et al., *Sci. Rep.* 2020 (`plissonf/ML-guided-...-non-hemolytic-peptides`), compiled from the Raghava-lab HemoPI / Hemolytik | **MIT** | **Yes** |

**Why HemoPI-2, not HemoPI-1.** HemoPI-1's negatives are random SwissProt fragments, so a model
trained on it learns "AMP-like ⇒ hemolytic" (it scored 99% of our APEX-active peptides as
hemolytic — useless for ranking selectivity *among* actives). HemoPI-2 discriminates high vs low
hemolytic potency between real peptides, giving a usable spread. Held-out AUROC 0.778 (moderate;
the harder, relevant task). Full rationale and the activity/selectivity trade-off in
`docs/RESEARCH.md`.
