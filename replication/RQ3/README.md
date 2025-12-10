# RQ3: Design Choices Ablation

Reproduces **Tables 4, 5, and 6** from the paper.

## Run

```bash
bash run_rq3.sh
```

All data is **pre-computed** (LLM filtering results, downstream metrics, model sweeps). No training or API calls needed.

## Structure

```
RQ3/
├── code/
│   └── generate_tables.py    # Table generation script
├── data/
│   ├── llm_filtering_results.csv         # Table 4 data
│   ├── label_variant_comparison.csv      # Table 5 data
│   └── experiment_results.csv            # Table 6 data
└── outputs/tables/                       # Output location
```

## Output

- `outputs/tables/table4_false_positive_detection.csv` → **Table 4** (Teacher LLM comparison)
- `outputs/tables/table5_downstream_performance.csv` → **Table 5** (Pseudo-labeler comparison)
- `outputs/tables/table6_model_scaling.csv` → **Table 6** (Model architecture comparison)

**Expected**: Claude-4 best teacher (F1=0.887), CodeT5p-220M best student (F1=0.794).
