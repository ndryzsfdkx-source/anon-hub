#!/bin/bash
set -euo pipefail

echo "========================================"
echo "RQ2: Cost-Effectiveness Evaluation"
echo "Reproducing Table 3 from the paper"
echo "========================================"
echo ""

# Navigate to the script directory
cd "$(dirname "$0")"

# Check prerequisites
if [ ! -d "inputs/oracle" ]; then
    echo "ERROR: Oracle dataset not found at inputs/oracle"
    exit 1
fi

if [ ! -d "inputs/ranking" ]; then
    echo "ERROR: Ranking data not found at inputs/ranking"
    exit 1
fi

echo "✓ Prerequisites check passed"
echo ""

# Run pipeline
echo "Running evaluation pipeline..."
echo ""

echo "[1/4] Building evaluation frames..."
python code/build_evalframe.py --config config/params.yaml

echo ""
echo "[2/4] Computing F1@N%LOC metrics..."
python code/compute_f1_at_n_loc.py --config config/params.yaml

echo ""
echo "[3/4] Computing Effort@N%Recall metrics..."
python code/compute_effort_at_n_recall.py --config config/params.yaml

echo ""
echo "[4/4] Rolling up results..."
python code/rollup_results.py --config config/params.yaml

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - outputs/tables/final/paper_main_results.csv  ← Table 3"
echo "  - outputs/tables/final/all_techs.csv"
echo ""
echo "Key metrics (IntelliSA):"
python3 -c "
import pandas as pd
df = pd.read_csv('outputs/tables/final/paper_main_results.csv')
intellisa = df[df['method'] == 'intellisa']
for _, row in intellisa.iterrows():
    print(f\"  {row['tech']:8s}: F1@1%LOC={row['f1_at_1pct']:.3f}, Effort@60%Recall={row['effort_at_60pct_recall']:.2f}%\")
"
echo ""

