#!/usr/bin/env bash
# ci/package.sh ARCH: target/release/runner -> dist/runner-linux-ARCH, .sha256, build-info-ARCH.json
set -euo pipefail
ARCH=$1
cd "$(dirname "$0")/.."
mkdir -p dist
B=runner-linux-$ARCH
cp target/release/runner "dist/$B"
strip "dist/$B" 2>/dev/null || true
( cd dist && sha256sum "$B" > "$B.sha256" )
GLIBC=$(objdump -T "dist/$B" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -n 1)
python3 - "dist/build-info-$ARCH.json" <<PY
import json, sys
json.dump({"arch": "$ARCH", "commit": "${GITHUB_SHA:-}", "rustc": "$(rustc --version)", "rustflags": "${RUSTFLAGS:-}",
           "glibc_max": "$GLIBC", "runner": "${RUNNER_NAME:-}", "sha256": open("dist/$B.sha256").read().split()[0],
           "built_at": "$(date -u +%FT%TZ)"}, open(sys.argv[1], "w"), indent=1)
PY
cat "dist/build-info-$ARCH.json"
