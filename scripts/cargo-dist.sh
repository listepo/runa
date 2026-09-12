#!/usr/bin/env bash
# Invoke the mise-pinned cargo-dist (0.28.0). ~/.cargo/bin may shadow with 0.32.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
prefix="${MISE_CARGO_DIST:-$HOME/.local/share/mise/installs/cargo-cargo-dist/0.28.0}"
bin="$prefix/bin/dist"
if [[ ! -x "$bin" ]]; then
  echo "missing $bin — run: mise install" >&2
  exit 1
fi
cd "$root"
exec "$bin" "$@"
