#!/bin/bash
set -e

echo "========================================"
echo "RQ1: Detection Accuracy Evaluation"
echo "Reproducing Table 2 from the paper"
echo "========================================"
echo ""

# Navigate to the script directory
cd "$(dirname "$0")"

# Check prerequisites
if [ ! -d "data/oracle" ]; then
    echo "ERROR: Oracle dataset not found at data/oracle"
    echo "Symlink may be broken. Please check data/oracle -> ../../datasets/oracle"
    exit 1
fi

if [ ! -d "data/baselines/glitch" ]; then
    echo "ERROR: GLITCH baseline detections not found"
    exit 1
fi

if [ ! -d "data/detections" ]; then
    echo "ERROR: Standardized detections not found"
    echo "Please ensure data/detections exists with method subdirectories"
    exit 1
fi

echo "✓ Prerequisites check passed"
echo ""

# Run evaluation
echo "Running evaluation script..."
echo ""

python code/evaluate.py \
    --root . \
    --use-detections-dir \
    --table-format full \
    --models "claude-sonnet-4.0-static-analysis-rules" \
    --models "claude-sonnet-4.0-definition-based" \
    --models "grok-4-fast-static-analysis-rules" \
    --models "grok-4-fast-definition-based" \
    --models "gpt-5-static-analysis-rules" \
    --models "gpt-5-2025-08-07-definition-based" \
    --suffix rq1_repro

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/metrics_ansible_rq1_repro.csv"
echo "  - results/metrics_chef_rq1_repro.csv"
echo "  - results/metrics_puppet_rq1_repro.csv"
echo "  - results/metrics_macro_rq1_repro.csv  ← Table 2"
echo ""
echo "View the macro results (Table 2):"
echo ""
cat results/metrics_macro_rq1_repro.csv
echo ""
echo "Expected IntelliSA Macro-F1: ~0.831"
echo ""

