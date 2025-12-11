# IntelliSA Replication Package

Reproduce experimental results from our paper.

## Setup

Install Python dependencies (Python 3.9+):

```bash
cd replication
pip install -r requirements.txt
```

## Run Experiments

```bash
bash RQ1/run_rq1.sh  # Table 2: Detection Accuracy
bash RQ2/run_rq2.sh  # Table 3: Cost-Effectiveness
bash RQ3/run_rq3.sh  # Tables 4-6: Design Choices
```

All inputs are **pre-computed**. No API calls or training needed. Total runtime: ~5 minutes.

## Results

| Paper Table | Output File                  | Location                    |
| ----------- | ---------------------------- | --------------------------- |
| Table 2     | metrics_macro_rq1_repro.csv  | `RQ1/results/`              |
| Table 3     | paper_main_results.csv       | `RQ2/outputs/tables/final/` |
| Table 4     | false_positive_detection.csv | `RQ3/outputs/tables/`       |
| Table 5     | downstream_performance.csv   | `RQ3/outputs/tables/`       |
| Table 6     | model_scaling.csv            | `RQ3/outputs/tables/`       |

Reference results are included. You can run scripts to regenerate or view provided results directly.

## Structure

```
replication/
├── datasets/         # Oracle test set (241 scripts, 213 smells)
│                     # Training data (2,300 instances)
├── RQ1/              # Detection accuracy evaluation
├── RQ2/              # Cost-effectiveness evaluation
├── RQ3/              # Design choices ablation
└── requirements.txt  # Shared dependencies
```

See individual `RQ*/README.md` for details.
