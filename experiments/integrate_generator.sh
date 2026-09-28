#!/usr/bin/env bash
# Promote a ReST-improved checkpoint to the shipped generator, keeping a backup.
# Usage: experiments/integrate_generator.sh experiments/rest/cN/best.pt
# After this, regenerate + run the official validator before committing.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="$1"
[ -f "$SRC" ] || { echo "no such checkpoint: $SRC"; exit 1; }

if [ ! -f checkpoint/generator_base.pt ]; then
  cp checkpoint/generator.pt checkpoint/generator_base.pt
  echo "backed up original -> checkpoint/generator_base.pt"
fi
cp "$SRC" checkpoint/generator.pt
echo "promoted $SRC -> checkpoint/generator.pt"
echo "next:"
echo "  uv run generate --n-sequences 500 --top-k 10 --out-dir /tmp/smoke   # quick check"
echo "  uv run python scripts/verify_submission.py <repo-url>               # full validator"
