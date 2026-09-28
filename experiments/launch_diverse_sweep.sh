#!/usr/bin/env bash
# Diversity-aware ReST: same loop, but the fine-tune set is de-duplicated by MinHash signature
# (--dedup-cap) so the generator is pushed to produce DIVERSE actives instead of collapsing onto
# a few high-reward motifs. Motivated by the round-5 sweep collapsing within-pool diversity.
# One chain per GPU. Config: gpu lambda gp mdr dedup seed
set -u
cd "$(dirname "$0")/.."
AGPU="$1"
COMMON="--init checkpoint/generator.pt --rounds 5 --n-raw 200000 --keep-n 12000 --epochs 2 --lr 1e-4 --anchor-frac 0.3 --gpu-python $AGPU"

CONFIGS=(
  "0 1.0 1.0 1.0 3 0"   # balanced + selective (c3-like) + diversity
  "1 0.5 1.0 1.0 3 0"   # balanced, higher activity (c5-like) + diversity
  "2 1.0 0.5 0.5 3 0"   # GN-strong + selective (c1-like) + diversity
  "3 1.5 1.0 1.0 3 0"   # more selective balanced + diversity
)

pids=()
for cfg in "${CONFIGS[@]}"; do
  read -r G LAM GP MDR DEDUP SEED <<< "$cfg"
  OUT="experiments/rest/d${G}"
  echo "launch d${G}: lambda=$LAM gp=$GP mdr=$MDR dedup=$DEDUP on GPU $G"
  uv run python experiments/rest_finetune.py $COMMON \
    --out-dir "$OUT" --gpu "$G" --gpus "$G" \
    --hemo-lambda "$LAM" --gp-w "$GP" --mdr-w "$MDR" --dedup-cap "$DEDUP" --seed "$SEED" \
    > "experiments/rest/d${G}.log" 2>&1 &
  pids+=($!)
  sleep 3
done
echo "diverse chains launched; waiting..."
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "DIVERSE_SWEEP_DONE (failures=$fail)"
