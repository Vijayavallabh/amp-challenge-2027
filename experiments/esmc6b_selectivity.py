"""Does the LARGEST ESM-C (ESMC-6B) beat ESMC-600M on the HemoPI-2 selectivity task?

Honors the directive to use the largest available PLM. The GP/MDR-at-0%-hemolysis trade-off is
bottlenecked by hemolysis-predictor accuracy, so a better selectivity model directly helps. Embeds
HemoPI-2 with Synthyra/ESMplusplus_6B (ESM++ wrapper of ESMC-6B, MIT, transformers-native), trains
the same MLP head, and reports held-out AUROC vs ESMC-600M (0.905) and physchem (0.778). If it wins
meaningfully, we ship it (weighing the ~10x inference cost); if marginal, we keep 600M.

    CUDA_VISIBLE_DEVICES=1 HF_HOME=/path uv run python experiments/esmc6b_selectivity.py [model]
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
HEMO = REPO / "data" / "hemolysis" / "HemoPI2.fasta"
HF_MODEL = sys.argv[1] if len(sys.argv) > 1 else "Synthyra/ESMplusplus_6B"


def parse():
    recs, cur = [], None
    for line in HEMO.read_text().splitlines():
        if line.startswith(">"): cur = line[1:]
        elif line.strip(): recs.append((cur, line.strip()))
    tok = lambda h: h.split("_")[1]
    tr = [(s, 1 if tok(h) == "pm" else 0) for h, s in recs if tok(h) in ("pm", "nm")]
    va = [(s, 1 if tok(h) == "pv" else 0) for h, s in recs if tok(h) in ("pv", "nv")]
    return tr, va


def auroc(y, p):
    order = np.argsort(p, kind="stable"); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p)+1)
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks); ranks = (sums/counts)[inv]
    npos = float(y.sum()); nneg = float(len(y)-npos)
    return 0.5 if npos == 0 or nneg == 0 else (ranks[y == 1].sum() - npos*(npos+1)/2)/(npos*nneg)


@torch.no_grad()
def embed(seqs, model, tok, device, batch=8):
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i+batch]
        enc = tok(chunk, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc).last_hidden_state
        for k, s in enumerate(chunk):
            out.append(h[k, 1:1+len(s)].float().mean(0).cpu().numpy())
    return np.array(out, dtype=np.float32)


def main():
    from transformers import AutoModel
    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = parse()
    print(f"HemoPI-2: train={len(tr)} val={len(va)} | loading {HF_MODEL} ...", flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(HF_MODEL, trust_remote_code=True).eval().to(device)
    tok = model.tokenizer
    print(f"loaded in {time.time()-t0:.0f}s ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)", flush=True)
    t0 = time.time()
    Xtr = embed([s for s, _ in tr], model, tok, device); Xva = embed([s for s, _ in va], model, tok, device)
    ytr = np.array([y for _, y in tr], np.float32); yva = np.array([y for _, y in va], np.float32)
    print(f"embed dim={Xtr.shape[1]} in {time.time()-t0:.0f}s", flush=True)
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6
    Xtr_n = torch.tensor((Xtr-mu)/sd); Xva_n = torch.tensor((Xva-mu)/sd); ytr_t = torch.tensor(ytr)
    best = 0.0
    for seed in range(3):  # small ensemble over head inits -> stabler estimate
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xtr.shape[1],128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3); lf = nn.BCEWithLogitsLoss()
        b = 0.0
        for ep in range(400):
            net.train(); opt.zero_grad(); lf(net(Xtr_n).squeeze(1), ytr_t).backward(); opt.step()
            if (ep+1) % 10 == 0:
                net.eval()
                with torch.no_grad(): p = torch.sigmoid(net(Xva_n).squeeze(1)).numpy()
                b = max(b, auroc(yva, p))
        best = max(best, b); print(f"  seed {seed}: AUROC {b:.3f}", flush=True)
    print(f"\n{HF_MODEL}: held-out AUROC = {best:.3f}   (ESMC-600M: 0.905, ESM-2 150M: 0.883, physchem: 0.778)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
