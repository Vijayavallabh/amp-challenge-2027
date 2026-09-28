"""Per-sub-model APEX prediction (anti-Goodhart check). Run with the CUDA torch 2.5.1 python
from the oracle/apex directory:

    CUDA_VISIBLE_DEVICES=0 <agpu-python> experiments/apex_permodel.py -i seqs.fasta -o out.npz -g 1

APEX ships an 8-member ensemble; the shipped scorer averages them. This emits every member
separately as an (n, 8, 11) array, so we can check whether ReST-optimized peptides are called
active by ALL sub-models (robust) or only a subset (a sign of overfitting to APEX's weights).
"""

import math
import os
import sys
from optparse import OptionParser

import numpy as np

APEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oracle", "apex")
sys.path.insert(0, APEX_DIR)

import torch  # noqa: E402
from APEX_models import AMP_model  # noqa: F401,E402  (referenced by the pickles)
from Bio import SeqIO  # noqa: E402
from utils import make_vocab, onehot_encoding  # noqa: E402

parser = OptionParser()
parser.add_option("-i", "--i", default="in.fasta")
parser.add_option("-o", "--o", default="permodel.npz")
parser.add_option("-g", "--g", default="1")
(opts, _) = parser.parse_args()

use_gpu = str(opts.g) == "1" and torch.cuda.is_available()
max_len = 52
word2idx, _ = make_vocab()

model_dir = os.path.join(APEX_DIR, "APEX_pathogen_models")
import glob  # noqa: E402
models = []
for p in sorted(glob.glob(os.path.join(model_dir, "APEX_*"))):
    m = torch.load(p, map_location="cpu").eval()
    models.append(m.cuda() if use_gpu else m.cpu())

seqs = [str(r.seq) for r in SeqIO.parse(open(opts.i), "fasta") if len(str(r.seq)) <= 50]
seqs = np.array(seqs)
bs = 3000
per_model = []  # each (n, 11)
for m in models:
    preds = []
    for i in range(int(math.ceil(len(seqs) / bs))):
        chunk = seqs[i * bs:(i + 1) * bs]
        rep = onehot_encoding(chunk, max_len, word2idx)
        X = torch.LongTensor(rep)
        X = X.cuda() if use_gpu else X
        with torch.no_grad():
            out = m(X).cpu().numpy()
        preds.append(10 ** (6 - out))
    per_model.append(np.vstack(preds))
arr = np.stack(per_model, axis=1)  # (n, 8, 11)
np.savez_compressed(opts.o, sequences=seqs.astype(object), permodel=arr.astype(np.float32))
print(f"per-model MIC saved: {arr.shape} -> {opts.o}")
