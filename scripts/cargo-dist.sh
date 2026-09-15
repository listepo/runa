#!/usr/bin/env bash
# Invoke the mise-pinned cargo-dist. ~/.cargo/bin may shadow with another
# version, and aqua-backend assets nest the binary one level deeper than the
# legacy `cargo:` layout, so probe the known layouts for the pinned version.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
ver="$(sed -n 's/.*"aqua:axodotdev\/cargo-dist" *= *"\([^"]*\)".*/\1/p' "$root/mise.toml" | head -1)"
base="$HOME/.local/share/mise/installs"
bin=""
for cand in \
    "$base/aqua-axodotdev-cargo-dist/$ver/cargo-dist-aarch64-apple-darwin/dist" \
    "$base/aqua-axodotdev-cargo-dist/$ver/bin/dist" \
    "$base/cargo-cargo-dist/$ver/bin/dist"; do
    if [[ -x "$cand" ]]; then
        bin="$cand"
        break
    fi
done
if [[ -z "$bin" ]]; then
    echo "missing mise-pinned cargo-dist $ver — run: mise install" >&2
    exit 1
fi
cd "$root"
exec "$bin" "$@"
