# AMP Challenge 2027 — submission

Generative design of antimicrobial peptides for the
[AMP Challenge 2027](https://szczurek-lab.github.io/amp-challenge-website/), a NeurIPS 2026
Competition Track challenge on discovering new antibiotics against drug-resistant bacteria.

`uv run generate` produces the two files the organizers validate: a 50,000-sequence library and a
ranked top-100 list, reproducibly from a fixed seed.

> **Status: activity-fine-tuned generator + wet-lab-calibrated composition ranking (feat-033).**
> `uv run generate` samples from an autoregressive Transformer (`checkpoint/generator.pt`) **fine-tuned
> toward predicted activity/selectivity by rejection sampling (ReST)** on the H100s (so its samples are
> mostly predicted-active, not the ~6% of the pre-trained base), then ranks the top-100 (from an 8×/400k
> oversampled pool) **not by APEX-predicted potency but by a wet-lab-calibrated composition score** —
> `lys_fraction − 0.5·aromatic_fraction` within an **APEX-active-band gate**, minus an **ESMC-600M
> selectivity** penalty (latest-SOTA protein LM, held-out AUROC 0.905), with a within-list diversity screen.
> We validated every ranking signal against *real* activity — our 46 wet-lab MICs **and** 946 independent
> DBAASP peptides — and found that inside the high-activity band the top-100 is drawn from, **APEX loses its
> ranking power** (Spearman +0.06) while **Lysine-richness predicts real activity** (+0.27); so APEX is kept
> only as the active-band gate and composition does the ranking (`--composition-weight 0` recovers the earlier
> feat-031 APEX ranking). The top-100 pool is drawn cooler (`--top-temperature 0.8`) — which supplies *more*
> Lysine-rich non-hemolytic candidates — while the library stays hot (temperature 1.6) for phase-1
> diversity/novelty. The assayed top-50 is a Lys/Arg **blend** (R/(R+K) 0.33, Lys>Arg in 94% vs 14% under
> APEX ranking), **0% predicted-hemolytic** (max P 0.04), and **novel** (top-50 median identity 0.69, top-100
> max 0.80). On every real-data test the composition selection ties-or-beats APEX selection on all four
> activity categories; its APEX-*predicted* profile looks lopsided (Broad 0.64 / Gram- 0.85 / Gram+ 0.26 /
> MDR 0.33) only because APEX mis-scores the Lysine chemotype our wet-lab + DBAASP data reward. All figures
> are computational predictions, not measurements. `--baseline` falls back if the oracle/checkpoint are
> unavailable. See [docs/RESEARCH.md](docs/RESEARCH.md) and [feature_list.json](feature_list.json).

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Nothing else — no pip, conda, or manual virtualenv.

```bash
git clone https://github.com/Vijayavallabh/amp-challenge-2027
cd amp-challenge-2027
./init.sh
```

`init.sh` installs from the lock file, runs the tests, generates the submission twice, and
confirms the two runs are byte-identical — the same path the organizers' validator takes.

To just generate:

```bash
uv sync
uv run generate
```

Output lands in `generate/`:

```
generate/
  library.fasta   50,000 unique peptides
  top.fasta       the ranked top 100, all drawn from the library
```

### Options

Every flag has a default, so a bare `uv run generate` is a complete run.

| Flag | Default | Description |
|---|---|---|
| `--n-sequences` | `50000` | Sequences in the library |
| `--top-k` | `100` | Sequences in the ranked top list |
| `--seed` | `42` | Random seed; fixed so runs are reproducible |
| `--min-length` | `8` | Shortest peptide to generate |
| `--max-length` | `50` | Longest peptide to generate |
| `--length` | _unset_ | Fix every peptide at exactly this length, overriding the two above |
| `--out-dir` | `generate` | Output directory |
| `--reference` | `data/antibacterial.fasta` | Known antibacterial peptides to screen against |
| `--checkpoint` | `checkpoint/generator.pt` | Trained generator weights |
| `--temperature` | `1.6` | Sampling temperature; hot sampling keeps the fine-tuned generator's library diverse/novel at no top-50 activity cost |
| `--top-p` | `1.0` | Nucleus sampling cutoff (trained generator) |
| `--oversample` | `8.0` | Pick the top-100 from this multiple of `--n-sequences` candidates (larger pool → stronger top list); `1.0` disables |
| `--rank` | `apex` | Top-100 ranking backbone: `apex` (APEX oracle, used as the active-band gate) or `likelihood` (generator) |
| `--composition-weight` | `1.0` | **Shipped ranking (feat-033):** within the APEX-active band, rank by `composition-weight·(lys_fraction − aromatic-weight·aromatic_fraction)`; `0` recovers the earlier APEX-potency ranking |
| `--aromatic-weight` | `0.5` | Aromatic (F/W/Y) penalty inside the composition score — aromatics anti-predict real activity and track hemolysis |
| `--rank-objective` | `balanced` | APEX active-band score: `balanced` (hard Gram+/MDR/Gram− Success Rate + broad tie-break), `category`, or `broad` (potency margin) |
| `--gp-weight` | `1.0` | Gram-positive up-weight in the APEX active-band objective |
| `--mdr-weight` | `1.0` | MDR up-weight in the APEX active-band objective |
| `--hemolysis-penalty` | `1.5` | Selectivity weight λ: subtract `λ·P(hemolytic)` (ESMC) from the rank score, keeping the top-50 non-hemolytic; `0` disables |
| `--apex-dir` | `oracle/apex` | APEX oracle project (isolated env), used when `--rank apex` |
| `--diversity-max-identity` | `0.6` | Cap pairwise identity within the top-100; `>=1` disables |
| `--baseline` | off | Force the random-baseline generator (no checkpoint / no torch needed) |
| `--skip-validation` | off | Write the files without the local compliance check |

Generation runs on a CUDA GPU when available and falls back to CPU. The full 50k library on CPU is
slow (tens of minutes); use a GPU for full runs, or `--baseline` / a small `--n-sequences` on CPU.

A quick run while developing:

```bash
uv run generate --n-sequences 500 --top-k 10 --out-dir /tmp/smoke
```

## Validation

The generator validates its own output and exits non-zero if anything fails. To run the
organizers' validator against the pushed public repository — it clones a fresh copy, syncs,
generates twice, and checks every rule:

```bash
uv run python scripts/verify_submission.py https://github.com/Vijayavallabh/amp-challenge-2027
```

Do this before submitting. There is one entry per model and no resubmission.

## Layout

```
├── AGENTS.md                       # working rules and invariants for contributors and agents
├── docs/COMPETITION.md             # the rules, deadlines, and evaluation criteria
├── docs/COMPLIANCE.md              # every rule, where it is enforced, how it is verified
├── docs/DATA.md                    # training data card: source, licence, composition, filters
├── SUBMISSION.md                   # the required write-up (abstract, data, ranking procedure)
├── init.sh                         # install, test, generate, check reproducibility
├── feature_list.json, progress.md  # what is done, what is next, with evidence
├── checkpoint/generator.pt         # trained generator weights (shipped)
├── data/antibacterial.fasta        # 39,448 known antibacterial peptides (reference set)
├── scripts/verify_submission.py    # the organizers' validator, vendored
├── tests/test_constraints.py       # tests over the compliance layer
└── src/amp_challenge_2027/
    ├── constraints.py              # the competition's hard rules as code
    ├── fasta.py                    # FASTA I/O matching the official parser
    ├── data.py                     # training corpus: loading, metadata, disclosure
    ├── paths.py                    # repo-root-aware path resolution
    ├── nn.py                       # the AR-Transformer architecture + deterministic sampling
    ├── model.py                    # PeptideGenerator: TrainedGenerator + RandomBaseline
    └── generate.py                 # the `generate` entry point
```

## Using a different model

`build_model()` in [`src/amp_challenge_2027/generate.py`](src/amp_challenge_2027/generate.py) is
the only place that needs to change. Implement the `PeptideGenerator` protocol:

```python
class MyModel:
    name = "my-model"

    def sample(self, n: int, rng: np.random.Generator) -> list[str]:
        """Return n candidate sequences. Duplicates and invalid ones are fine —
        the caller filters them."""

    def score(self, sequences: list[str]) -> list[float]:
        """One score per sequence, higher is better. This builds the ranked top 100."""
```

Then return it from `build_model()`. De-duplication, constraint filtering, the novelty screen,
ranking, and file writing all stay as they are, so swapping models cannot break the submission
contract.

Load weights from `checkpoint/` using `resolve_repo_path()`, which works regardless of the
working directory the validator runs from.

Two things to get right, both checked by the validator:

- **Determinism.** Seed everything from `--seed`, including framework RNGs. Two runs are compared
  byte for byte. `AGENTS.md` lists the traps that break this.
- **Ranking.** `score()` is what the competition actually measures — 25 peptides are drawn at
  random from your top 50 and assayed. An arbitrary ranking wastes the entry.

Dependencies go in with `uv add <package>`; commit the updated `uv.lock`.

## Licence

MIT — see [LICENSE](LICENSE). A permissive OSI-approved licence is a condition of co-authorship
eligibility.

`scripts/verify_submission.py` and `data/antibacterial.fasta` are vendored from
[szczurek-lab/amp-challenge-2027](https://github.com/szczurek-lab/amp-challenge-2027), which is
BSD-3-Clause; that licence is retained at `data/UPSTREAM_TEMPLATE_LICENSE.txt`.
