# Hemolysis data (selectivity oracle)

The **shipped selectivity model** (`checkpoint/hemolysis.pt`) is trained on **`HemoPI2.fasta`**
(high vs low hemolytic potency, both real Hemolytik peptides) — see the note at the bottom for
why, not `HemoPI1.fasta`. `HemoPI1.fasta` is retained for reference/disclosure. Both use the same
train/held-out-validation split encoded in the FASTA ids:

| id token | label | split |
|---|---|---|
| `peptide_pm_*` | hemolytic | train (main) |
| `peptide_nm_*` | non-hemolytic | train (main) |
| `peptide_pv_*` | hemolytic | held-out validation |
| `peptide_nv_*` | non-hemolytic | held-out validation |

**Provenance.** Vendored from
[plissonf/ML-guided-discovery-and-design-of-non-hemolytic-peptides](https://github.com/plissonf/ML-guided-discovery-and-design-of-non-hemolytic-peptides)
(Plisson, Ramírez-Sánchez, Martínez-Hernández, *Sci. Rep.* 2020, "Machine learning-guided
discovery and design of non-hemolytic peptides"), which is **MIT-licensed** — hence
redistributable here. That repository in turn compiled HemoPI-1 from the Raghava-lab HemoPI
resource (Chaudhary et al. 2016), itself built on the Hemolytik database — all public.

Used only to **train the shipped hemolysis/selectivity classifier** (`checkpoint/hemolysis.pt`),
not as generator training data. Disclosed in `docs/DATA.md`.

**HemoPI-2, not HemoPI-1.** HemoPI-1's negatives are random SwissProt fragments, so a classifier
trained on it learns "AMP-like ⇒ hemolytic" and cannot rank hemolysis among active AMPs (verified:
it scored 99% of our APEX-active peptides as hemolytic). HemoPI-2 discriminates the *degree* of
hemolysis between real peptides — the signal we actually need. See `docs/RESEARCH.md`.
