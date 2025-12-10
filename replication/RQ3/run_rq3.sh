#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "========================================"
echo "RQ3: Design Choices Ablation"
echo "========================================"
echo ""

python3 code/generate_tables.py

