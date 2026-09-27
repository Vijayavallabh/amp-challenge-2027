#!/usr/bin/env bash
# Startup and verification gate for the AMP Challenge 2027 submission repo.
#
# Mirrors what the organizers do (uv sync, then run the entry point twice and compare)
# so a green run here means the submission would survive their validator.
#
# Read-only apart from regenerating the gitignored generate/ directory.

set -euo pipefail

cd "$(dirname "$0")"

fail() { echo; echo "FAILED: $1" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || fail "uv is not on PATH. Install it: https://docs.astral.sh/uv/"

echo "=== [1/4] uv sync --locked ==="
uv sync --locked || fail "dependency install. If you changed pyproject.toml, run 'uv lock' and commit uv.lock."

echo
echo "=== [2/4] uv run pytest -q ==="
uv run pytest -q || fail "test suite"

echo
echo "=== [3/4] uv run generate (first pass) ==="
uv run generate || fail "generation, or the built-in compliance check"

first_library=$(md5sum generate/library.fasta | cut -d' ' -f1)
first_top=$(md5sum generate/top.fasta | cut -d' ' -f1)

echo
echo "=== [4/4] uv run generate (second pass, reproducibility) ==="
uv run generate >/dev/null || fail "generation on the second pass"

second_library=$(md5sum generate/library.fasta | cut -d' ' -f1)
second_top=$(md5sum generate/top.fasta | cut -d' ' -f1)

if [ "$first_library" != "$second_library" ] || [ "$first_top" != "$second_top" ]; then
  echo "  library: $first_library -> $second_library"
  echo "  top:     $first_top -> $second_top"
  fail "two runs produced different output. See 'Reproducibility Traps' in AGENTS.md."
fi

echo "  library.fasta $first_library (stable)"
echo "  top.fasta     $first_top (stable)"

cat <<EOF

=== Verification complete ===

The submission is structurally valid and reproducible.

Next steps:
1. Read feature_list.json and pick ONE feature that is not done.
2. Implement only that feature.
3. Re-run ./init.sh before claiming it is done, and record the evidence.

To run the organizers' own validator against the pushed public repo:
  uv run python scripts/verify_submission.py <github-url>
EOF
