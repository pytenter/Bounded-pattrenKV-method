#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "AIME24 pseudo-decode 3090 status"
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
find reports/aime24_pseudodecode_3090_8gpu -maxdepth 1 -type f -printf '%f %s bytes\n' | sort || true
nvidia-smi
