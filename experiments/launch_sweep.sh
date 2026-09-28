#!/usr/bin/env bash
# Parallel ReST sweep: one chain per H100, mapping the activity/selectivity (lambda) and
# Gram+/MDR up-weight (gp/mdr) tradeoffs. Each chain is self-contained on its own GPU
# (sample + APEX score + fine-tune all on GPU $G), so all 8 run concurrently.
set -u
cd "$(dirname "$0")/.." || exit 1
AGPU="$1"   # path to the CUDA torch 2.5.1 python
COMMON="--init checkpoint/generator.pt --rounds 5 --n-raw 150000 --keep-n 12000 --epochs 2 --lr 1e-4 --gpu-python $AGPU"

# config: gpu lambda gp mdr seed
CONFIGS=(
  "0 0.5 0.5 0.5 0"
  "1 1.0 0.5 0.5 0"
  "2 2.0 0.5 0.5 0"
  "3 1.0 1.0 1.0 0"
  "4 1.0 0.0 0.0 0"
  "5 0.5 1.0 1.0 0"
  "6 1.5 1.0 1.0 0"
  "7 1.0 0.5 0.5 1"
)

pids=()
for cfg in "${CONFIGS[@]}"; do
  read -r G LAM GP MDR SEED <<< "$cfg"
  OUT="experiments/rest/c${G}"
  echo "launch chain c${G}: lambda=$LAM gp=$GP mdr=$MDR seed=$SEED on GPU $G"
  uv run python experiments/rest_finetune.py $COMMON \
    --out-dir "$OUT" --gpu "$G" --gpus "$G" \
    --hemo-lambda "$LAM" --gp-w "$GP" --mdr-w "$MDR" --seed "$SEED" \
    > "experiments/rest/c${G}.log" 2>&1 &
  pids+=($!)
  sleep 3   # stagger checkpoint loads
done

echo "all 8 chains launched; waiting..."
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail+1))
done
echo "SWEEP DONE (failures=$fail)"
