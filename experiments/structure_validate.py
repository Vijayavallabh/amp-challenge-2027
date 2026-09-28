"""Offline structure-based validation of the shipped top-100 with ESMFold2 (latest SOTA).

Folds peptides with **ESMFold2-Fast** (biohub/ESMFold2-Fast, ESMC-6B backbone, MIT) -- the latest
SOTA single-sequence structure predictor -- and reports, per peptide:
  * mean pLDDT  (0-1; the model's per-residue confidence)
  * helix fraction  (from backbone phi/psi dihedrals of the predicted all-atom structure)
  * Eisenberg hydrophobic moment  (sequence-based amphipathicity, assumes an alpha-helix)

This is an ORTHOGONAL, offline anti-Goodhart cross-check: active AMPs act by folding into
amphipathic alpha-helices that insert into bacterial membranes, so a peptide the sequence oracles
rate active *should* fold into a confident helix. It does NOT touch the submitted files (ESMFold2 is
a diffusion model and non-deterministic), so the shipped submission stays byte-reproducible; a fixed
seed only makes these reported numbers stable. pTM is deliberately not used as a primary signal --
it is calibrated for domains and is uninformative for 12-40-mers (melittin scores ptm~0.29).

    CUDA_VISIBLE_DEVICES=1 HF_HOME=/path uv run python experiments/structure_validate.py [fasta ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "experiments" / "cache"
HF_MODEL = "biohub/ESMFold2-Fast"

# Eisenberg consensus hydrophobicity scale (kcal/mol-ish, higher = more hydrophobic).
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85, "E": -0.74,
    "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19,
    "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}


def hydrophobic_moment(seq: str, angle: float = 100.0) -> float:
    """Eisenberg hydrophobic moment for an ideal alpha-helix (per-residue, delta=100 deg)."""
    h = np.array([EISENBERG.get(a, 0.0) for a in seq])
    theta = np.deg2rad(angle) * np.arange(len(seq))
    mx = float(np.sum(h * np.cos(theta)))
    my = float(np.sum(h * np.sin(theta)))
    return float(np.sqrt(mx * mx + my * my) / max(len(seq), 1))


def _dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-8)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def parse_backbone(pdb: str):
    """Return per-residue (N, CA, C) coords and CA B-factor (=pLDDT*100) from a PDB string."""
    res: dict[int, dict] = {}
    for line in pdb.splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name not in ("N", "CA", "C"):
            continue
        resseq = int(line[22:26])
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        d = res.setdefault(resseq, {})
        d[name] = xyz
        if name == "CA":
            d["b"] = float(line[60:66])
    order = sorted(res)
    N = np.array([res[i]["N"] for i in order if {"N", "CA", "C"} <= res[i].keys()])
    CA = np.array([res[i]["CA"] for i in order if {"N", "CA", "C"} <= res[i].keys()])
    C = np.array([res[i]["C"] for i in order if {"N", "CA", "C"} <= res[i].keys()])
    b = np.array([res[i].get("b", np.nan) for i in order if {"N", "CA", "C"} <= res[i].keys()])
    return N, CA, C, b


def helix_fraction(N, CA, C) -> float:
    """Fraction of residues in the alpha/3-10 helix (phi,psi) basin (generous DSSP-lite box)."""
    n = len(CA)
    if n < 4:
        return 0.0
    helix = 0
    counted = 0
    for i in range(1, n - 1):
        phi = _dihedral(C[i - 1], N[i], CA[i], C[i])
        psi = _dihedral(N[i], CA[i], C[i], N[i + 1])
        counted += 1
        if -145.0 <= phi <= -35.0 and -70.0 <= psi <= 25.0:
            helix += 1
    return helix / max(counted, 1)


def read_fasta(p: Path) -> list[tuple[str, str]]:
    out, name = [], None
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
        elif line:
            out.append((name or f"seq{len(out)}", line))
    return out


CONTROLS = [
    ("melittin(+helix)", "GIGAVLKVLTTGLPALISWIKRKRQQ"),
    ("magainin2(+helix)", "GIGKFLHSAKKFGKAFVGEIMNS"),
    ("aurein1.2(+helix)", "GLFDIIKKIAESF"),
    ("LL37frag(+helix)", "FKRIVQRIKDFLRNLV"),
    ("polyG(-)", "GGGGGGGGGGGGGGGGGG"),
    ("polyGS(-)", "GSGSGSGSGSGSGSGSGS"),
]


def main() -> int:
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2Model

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {HF_MODEL} ...", flush=True)
    model = EsmFold2Model.from_pretrained(HF_MODEL).eval().to(dev)

    def analyze(seq: str) -> dict:
        torch.manual_seed(0)  # stable reported numbers (diffusion sampler)
        pdb = model.infer_protein_as_pdb(seq, num_loops=3, num_sampling_steps=50)
        N, CA, C, b = parse_backbone(pdb)
        return {
            "len": len(seq),
            "plddt": float(np.nanmean(b)) if len(b) else float("nan"),  # B-factor already 0-1
            "helix": helix_fraction(N, CA, C),
            "mu_H": hydrophobic_moment(seq),
        }

    # Load sets: controls always; then any fasta args (default: shipped top-100).
    fastas = [Path(a) for a in sys.argv[1:]] or [REPO / "generate" / "top.fasta"]

    print("\n=== controls (validate the metric) ===")
    print(f"{'name':20s} {'len':>3s} {'pLDDT':>6s} {'helix':>6s} {'muH':>5s}")
    for name, seq in CONTROLS:
        r = analyze(seq)
        print(f"{name:20s} {r['len']:3d} {r['plddt']:6.3f} {r['helix']:6.2f} {r['mu_H']:5.2f}", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)

    # Length-matched random-peptide contrast: fold uniform-random valid peptides at the same
    # lengths as the primary set, so the top-100's fold quality is judged *relative* to noise
    # (pLDDT/helix are poorly calibrated in absolute terms for short peptides).
    primary = next((f for f in fastas if f.exists()), None)
    if primary is not None:
        rng = np.random.default_rng(0)
        lens = [len(s) for _, s in read_fasta(primary)]
        n_ctrl = min(40, len(lens))
        pick = rng.choice(len(lens), size=n_ctrl, replace=False)
        rand = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=lens[i])) for i in pick]
        rr = [analyze(s) for s in rand]
        rpl = np.array([r["plddt"] for r in rr]); rhx = np.array([r["helix"] for r in rr])
        print(f"\n=== random-peptide contrast (n={n_ctrl}, length-matched) ===")
        print(f"  mean pLDDT : median={np.median(rpl):.3f}   helix frac : median={np.median(rhx):.2f} "
              f"frac>0.5={(rhx>0.5).mean():.2f}")

    for fa in fastas:
        if not fa.exists():
            print(f"\n(skip {fa}: not found)")
            continue
        seqs = read_fasta(fa)
        rows = []
        for _, s in seqs:
            rows.append(analyze(s))
        pl = np.array([r["plddt"] for r in rows])
        hx = np.array([r["helix"] for r in rows])
        mu = np.array([r["mu_H"] for r in rows])
        print(f"\n=== {fa}  (n={len(rows)}) ===")
        print(f"  mean pLDDT : median={np.median(pl):.3f} mean={pl.mean():.3f} "
              f"frac<0.5={(pl<0.5).mean():.2f}")
        print(f"  helix frac : median={np.median(hx):.2f} mean={hx.mean():.2f} "
              f"frac>0.5={(hx>0.5).mean():.2f} frac<0.2={(hx<0.2).mean():.2f}")
        print(f"  muH (amphi): median={np.median(mu):.2f} mean={mu.mean():.2f}")
        # anti-Goodhart flags: confident? helical? amphipathic?
        weak = int(((pl < 0.55) & (hx < 0.3)).sum())
        print(f"  flagged (low pLDDT<0.55 AND low helix<0.3): {weak}/{len(rows)}")
        # save per-peptide
        out = CACHE / (fa.stem + "_structure.csv")
        with out.open("w") as f:
            f.write("seq,len,plddt,helix,muH\n")
            for (name, s), r in zip(seqs, rows):
                f.write(f"{s},{r['len']},{r['plddt']:.4f},{r['helix']:.4f},{r['mu_H']:.4f}\n")
        print(f"  per-peptide -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
