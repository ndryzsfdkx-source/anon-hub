# RQ2: Cost-Effectiveness

Reproduces **Table 3** from the paper.

## Run

```bash
bash run_rq2.sh
```

All inputs are **pre-computed** (ranked detections with confidence scores for 9 methods). No API calls needed.

## Structure

```
RQ2/
├── code/                # Evaluation pipeline scripts
├── config/
│   └── params.yaml      # Metrics configuration
├── inputs/
│   ├── oracle/          # Test set (symlink to ../../datasets/oracle/)
│   └── ranking/         # Pre-ranked detections with confidence scores
└── outputs/
    └── tables/final/    # Output location
```

## Output

- `outputs/tables/final/paper_main_results.csv` → **Table 3**
- Key metrics: `f1_at_1pct` (F1@1%LOC) and `effort_at_60pct_recall` (Effort@60%Recall)

**Expected**: IntelliSA achieves best F1@1%LOC (85%, 55%, 65%) and Effort@60%Recall (0.74%, 1.66%, 1.14%) on Ansible, Chef, Puppet.
