# AMP Challenge 2027 — submission

Generative design of antimicrobial peptides for the
[AMP Challenge 2027](https://szczurek-lab.github.io/amp-challenge-website/), a NeurIPS 2026
Competition Track challenge on discovering new antibiotics against drug-resistant bacteria.

`uv run generate` produces the two files the organizers validate: a 50,000-sequence library and a
ranked top-100 list, reproducibly from a fixed seed.

> **Status: trained generator + APEX activity ranking.** `uv run generate` samples from a trained
> autoregressive Transformer (`checkpoint/generator.pt`) — novel, cationic/amphipathic peptides —
> and ranks the top-100 by **APEX**-predicted MIC across the 11-pathogen panel (the wet-lab-aligned
> oracle, run as an isolated subprocess; see [docs/RESEARCH.md](docs/RESEARCH.md)), with a
> within-list diversity screen. This replaced model-likelihood ranking, which sat at only the ~30th
> percentile of predicted potency. Selectivity (hemolysis) is next (feat-013). `--rank likelihood`
> or `--baseline` fall back if the oracle/checkpoint are unavailable. See
> [feature_list.json](feature_list.json).

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
| `--temperature` | `1.0` | Sampling temperature (trained generator) |
| `--top-p` | `1.0` | Nucleus sampling cutoff (trained generator) |
| `--rank` | `apex` | Top-100 ranking: `apex` (predicted MIC) or `likelihood` (generator) |
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
