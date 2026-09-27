# Experimentally characterized AMP-Diffusion peptides

> **These are not the candidates in
> [`generate_broad_spectrum/top.fasta`](../generate_broad_spectrum/top.fasta).** This is
> AMP-Diffusion's *wet-lab* track record: peptides that were synthesized and assayed in the
> AMP-Diffusion study (Torres et al., *Cell Biomaterials* 2025). The challenge asks each
> baseline to ship "the complete set of experimentally characterized peptides ... with their
> measured MIC values" (AMP Challenge §1.6), and that is what this is. The generated top-100
> is a separate, freshly generated set of *new, unsynthesized* candidates; the challenge's
> novelty rule (<80% identity to known AMPs) keeps the two sets disjoint.

## `mic.csv`

Experimentally **measured** minimum inhibitory concentrations, long format — one row per
(peptide, strain) measurement:

| Column | Description |
|--------|-------------|
| `peptide_id` | AMP-Diffusion peptide id (`AMP_diff2-*`) |
| `sequence` | amino-acid sequence (canonical AAs) |
| `modification` | terminal/other modifications, if any |
| `strain` | `Genus species STRAIN_ID` (with resistance markers) |
| `strain_type` | `gram+` / `gram-` |
| `mic` | measured MIC value |
| `mic_unit` | `uM` |
| `mic_relation` | `=` exact, `>` right-censored (assay ceiling) |
| `medium`, `cfu_per_ml`, `ph`, `salt` | assay conditions |

All **46** synthesized peptides were tested against the full panel of **11** clinically
relevant pathogens (including MDR ESKAPE members) by broth microdilution — 506 (peptide, strain)
rows. A `mic_relation` of `>` marks a right-censored result: the peptide did not inhibit that
strain up to the 64 µM assay ceiling (`mic = 64`, `>`). Every pair was tested, so `>64` means
*no inhibition*, not *not measured*. See Torres et al. 2025 for full assay conditions.

## Provenance

Data extracted from Torres et al., "Generative latent diffusion language modeling yields
anti-infective synthetic peptides," *Cell Biomaterials* 1, 100183 (2025),
[doi:10.1016/j.celbio.2025.100183](https://doi.org/10.1016/j.celbio.2025.100183) (CC BY 4.0),
via the curated `data/validated/` set of the AMP generative benchmark.
