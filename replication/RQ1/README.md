# RQ1: Detection Accuracy

Reproduces **Table 2** from the paper.

## Run

```bash
bash run_rq1.sh
```

All detections are **pre-computed** (GLITCH, SLAC/SLIC, 6 LLM baselines, IntelliSA). No API calls needed.

## Structure

```
RQ1/
├── code/                # Evaluation scripts
│   └── evaluate.py      # Main script (produces Table 2)
├── data/
│   ├── oracle/          # Test set (symlink to ../../datasets/oracle/)
│   ├── baselines/       # Raw GLITCH/SLAC/SLIC detections
│   ├── intellisa/       # Raw IntelliSA detections
│   └── detections/      # Standardized format for evaluation
└── results/             # Output location
```

## Output

- `results/metrics_macro_rq1_repro.csv` → **Table 2** (Macro-F1 scores)
- `results/metrics_{tech}_rq1_repro.csv` → Per-technology details

**Expected**: IntelliSA Macro-F1 = 0.831
