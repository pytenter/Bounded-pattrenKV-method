#!/usr/bin/env bash
set -euo pipefail

nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
which nvcc || true
nvcc --version || true
gcc --version | head -n 1
g++ --version | head -n 1
/data/zypan/kvarn-repro/tools/bin/micromamba env list || true
df -h
free -h
env | grep -i proxy || true
python --version

