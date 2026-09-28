#!/usr/bin/env bash
# Evaluate a generator checkpoint's REAL shipped top-100 (diversity-capped), by sampling a
# large pool, GPU-scoring it, and running the exact shipped selection under two objectives.
# Usage: experiments/eval_checkpoint.sh <ckpt> <tag> <agpu-python> [n_raw]
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="$1"; TAG="$2"; AGPU="$3"; NRAW="${4:-300000}"
uv run python experiments/sample_pool.py --checkpoint "$CKPT" --n-raw "$NRAW" \
  --novel-only --seed 7 --gpu 0 --out "experiments/cache/${TAG}.txt" 2>&1 | grep -vE 'warning|Warning' | tail -1
uv run --no-project python experiments/bulk_apex.py --in "experiments/cache/${TAG}.txt" \
  --out "experiments/cache/${TAG}.npz" --device cuda --gpu-python "$AGPU" --gpus 0,1,2,3,4,5,6,7 2>&1 | tail -1
echo "[$TAG] shipped objective broad_potency-2*hemo, div 0.6:"
uv run python experiments/preview_select.py --cache "experiments/cache/${TAG}.npz" \
  --objective broad --hemo-lambda 2.0 --diversity 0.6 2>&1 | grep -vE 'warning|Warning'
echo "[$TAG] reward objective (gp/mdr upweight, lambda 1), div 0.6:"
uv run python experiments/preview_select.py --cache "experiments/cache/${TAG}.npz" \
  --objective reward --hemo-lambda 1.0 --diversity 0.6 2>&1 | grep -vE 'warning|Warning'
