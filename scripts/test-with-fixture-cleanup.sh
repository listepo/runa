#!/usr/bin/env bash
# Run the test suite, then clean downloaded fixtures only on a full pass.
#
#   scripts/test-with-fixture-cleanup.sh [-- <cargo test args...>]
#
# Cleanup removes git-ignored weights from tests/fixtures (keeps hand-made
# files: synthetic-*.gguf, audio/, video/, api/). Set RUNA_KEEP_FIXTURES=1
# (or KEEP=1) to skip cleanup locally and avoid re-downloading weights.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo test "$@"
status=$?

if [ "$status" -ne 0 ]; then
  echo "tests failed (exit $status); keeping fixtures for debugging" >&2
  exit "$status"
fi

if [ "${RUNA_KEEP_FIXTURES:-${KEEP:-0}}" = "1" ]; then
  echo "tests passed; RUNA_KEEP_FIXTURES=1, keeping fixtures"
  exit 0
fi

echo "tests passed; cleaning downloaded fixtures"
python3 scripts/clean-fixtures.py
