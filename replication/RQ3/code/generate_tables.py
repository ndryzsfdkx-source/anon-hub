#!/usr/bin/env python3
"""Generate the RQ3 replication tables for the IntelliSA paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "tables"


# ----------------------------- helpers ------------------------------------ #


def _format_with_std(mean: float, std: float) -> str:
    """Return a compact mean±std string rounded to three decimals."""
    return f"{mean:.3f}±{std:.3f}"


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


# Model name mappings to match paper display names
MODEL_NAME_MAPPING = {
    "Claude-Sonnet-4": "Claude-4",
    "GPT-5-Chat": "GPT-5",
    "Grok-4-Fast": "Grok-4",
}


def _normalize_model_name(name: str) -> str:
    """Map internal model names to paper display names."""
    return MODEL_NAME_MAPPING.get(name, name)


# --------------------------- Table 4 (LLM) -------------------------------- #


def generate_table4_false_positive_detection() -> pd.DataFrame:
    """Aggregate false-positive detection metrics for the teacher LLMs.

    Source: data/llm_filtering_results.csv (per-smell filtering outcomes).
    The table matches paper Table 4: precision/recall/F1 when treating
    GLITCH false positives as the positive class.
    """

    df = pd.read_csv(DATA_DIR / "llm_filtering_results.csv")

    rows: List[Dict[str, float]] = []
    for model_name, group in df.groupby("Model_Name"):
        total_fp = group["GLITCH_FP"].sum()
        filtered_fp = group["Filtered_FP"].sum()
        lost_tp = group["Lost_TP"].sum()

        precision = (
            filtered_fp / (filtered_fp + lost_tp) if (filtered_fp + lost_tp) > 0 else 0.0
        )
        recall = filtered_fp / total_fp if total_fp > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "Teacher Model": _normalize_model_name(model_name),
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Total GLITCH FP": total_fp,
                "Filtered FP": filtered_fp,
                "Lost TP": lost_tp,
            }
        )

    order = ["GPT-5", "Grok-4", "Claude-4"]
    summary = pd.DataFrame(rows)
    summary["Teacher Model"] = pd.Categorical(summary["Teacher Model"], order)
    summary = summary.sort_values("Teacher Model").reset_index(drop=True)

    formatted = summary.copy()
    for col in ("Precision", "Recall", "F1"):
        formatted[col] = formatted[col].apply(_format_metric)
    formatted.to_csv(OUTPUT_DIR / "table4_false_positive_detection.csv", index=False)
    return summary


# --------------------------- Table 5 (student) ---------------------------- #


LABEL_MAPPING = {
    "label1": "Claude-4",
    "label2": "Grok-4",
    "label3": "GPT-5",
}

# Column indices in the raw spreadsheet for each metric/label variant.
METRIC_COLUMN_INDEXES = {
    "f1": {"label1": 5, "label2": 6, "label3": 7},
    "precision": {"label1": 8, "label2": 9, "label3": 10},
    "recall": {"label1": 11, "label2": 12, "label3": 13},
}


@dataclass
class MetricSummary:
    labeler: str
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float

    @property
    def as_dict(self) -> Dict[str, float]:
        return {
            "Labeler": self.labeler,
            "Precision_Mean": self.precision_mean,
            "Precision_Std": self.precision_std,
            "Recall_Mean": self.recall_mean,
            "Recall_Std": self.recall_std,
            "F1_Mean": self.f1_mean,
            "F1_Std": self.f1_std,
            "Precision": _format_with_std(self.precision_mean, self.precision_std),
            "Recall": _format_with_std(self.recall_mean, self.recall_std),
            "F1": _format_with_std(self.f1_mean, self.f1_std),
        }


def generate_table5_downstream_performance() -> pd.DataFrame:
    """Summarize downstream detection after retraining on each pseudo-labeler.

    Mirrors the logic in `12.pseudo-label-quality/generate_rq31_artifacts.py`
    to reproduce paper Table 5.
    """

    raw = pd.read_csv(DATA_DIR / "label_variant_comparison.csv", header=[0, 1])
    df_data = raw.iloc[1:].copy()  # drop duplicated header row

    downstream = pd.DataFrame(
        {"test_set": df_data.iloc[:, 0].values, "seed": df_data.iloc[:, 1].values}
    )

    for metric_name, column_map in METRIC_COLUMN_INDEXES.items():
        for label_key, col_idx in column_map.items():
            downstream[f"{metric_name}_{label_key}"] = pd.to_numeric(
                df_data.iloc[:, col_idx], errors="coerce"
            )

    combined = downstream[downstream["test_set"] == "combined"]

    summaries: List[MetricSummary] = []
    for label_key, labeler_name in LABEL_MAPPING.items():
        precision = combined[f"precision_{label_key}"]
        recall = combined[f"recall_{label_key}"]
        f1 = combined[f"f1_{label_key}"]
        summaries.append(
            MetricSummary(
                labeler=labeler_name,
                precision_mean=precision.mean(),
                precision_std=precision.std(ddof=0),
                recall_mean=recall.mean(),
                recall_std=recall.std(ddof=0),
                f1_mean=f1.mean(),
                f1_std=f1.std(ddof=0),
            )
        )

    table = pd.DataFrame([s.as_dict for s in summaries])
    order = ["GPT-5", "Grok-4", "Claude-4"]
    table["Labeler"] = pd.Categorical(table["Labeler"], order)
    table = table.sort_values("Labeler").reset_index(drop=True)

    table.to_csv(OUTPUT_DIR / "table5_downstream_performance.csv", index=False)
    return table


# --------------------------- Table 6 (models) ----------------------------- #


PARAMS_BY_MODEL: Dict[str, int] = {
    "codet5_small": 60,
    "codebert_base": 110,
    "codet5_base": 220,
    "codet5_large": 770,
    "unixcoder_base": 110,
}

DISPLAY_NAMES: Dict[str, str] = {
    "codet5p_220m": "CodeT5p-220M",
    "codet5p_770m": "CodeT5p-770M",
    "codebert_base": "CodeBERT-base",
    "unixcoder_base": "UniXcoder-base",
    "codet5_base": "CodeT5-base",
    "codet5_large": "CodeT5-large",
    "codet5_small": "CodeT5-small",
}


def _params_from_row(row: pd.Series) -> int:
    size = str(row["size"]).strip().lower()
    if size.endswith("m"):
        try:
            return int(float(size[:-1]))
        except ValueError:
            pass
    model = row["model_name"]
    if model in PARAMS_BY_MODEL:
        return PARAMS_BY_MODEL[model]
    raise ValueError(f"Missing parameter mapping for model '{model}'")


def _metric_key(row: pd.Series) -> tuple[float, float, float, float]:
    return (row["f1"], row["accuracy"], row["precision"], row["recall"])


def generate_table6_model_scaling() -> pd.DataFrame:
    """Pick the best checkpoint per model and show macro-F1 vs parameter count."""

    experiments = pd.read_csv(DATA_DIR / "experiment_results.csv")

    # Best checkpoint per model_name by F1 (with accuracy/precision/recall tiebreaks).
    best_rows: Dict[str, pd.Series] = {}
    for _, row in experiments.iterrows():
        model = row["model_name"]
        current = best_rows.get(model)
        if current is None or _metric_key(row) > _metric_key(current):
            best_rows[model] = row

    records = []
    desired_models = [
        "codebert_base",
        "unixcoder_base",
        "codet5_base",
        "codet5p_220m",
        "codet5p_770m",
    ]
    for model_name in desired_models:
        if model_name not in best_rows:
            raise KeyError(f"Model '{model_name}' not found in experiment results.")
        row = best_rows[model_name]
        params = _params_from_row(row)
        display_name = DISPLAY_NAMES.get(model_name, model_name)
        records.append(
            {
                "Student Model": display_name,
                "Params (M)": params,
                "Macro F1 (argmax)": float(row["f1"]),
            }
        )

    table = pd.DataFrame(records)
    table = table.sort_values(["Params (M)", "Macro F1 (argmax)"], ascending=[True, False]).reset_index(
        drop=True
    )
    table.to_csv(OUTPUT_DIR / "table6_model_scaling.csv", index=False)
    return table


# ------------------------------ entrypoint -------------------------------- #


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("Generating RQ3 artifacts into:", OUTPUT_DIR)
    print()

    table4 = generate_table4_false_positive_detection()
    print("✓ Table 4 (False Positive Detection)")
    print(table4[["Teacher Model", "Precision", "Recall", "F1"]].to_string(index=False))
    print()

    table5 = generate_table5_downstream_performance()
    print("✓ Table 5 (Downstream Performance)")
    print(table5[["Labeler", "Precision", "Recall", "F1"]].to_string(index=False))
    print()

    table6 = generate_table6_model_scaling()
    print("✓ Table 6 (Model Scaling)")
    print(table6.to_string(index=False))
    print()

    print("========================================")
    print("Tables saved to outputs/tables/:")
    print("  - table4_false_positive_detection.csv")
    print("  - table5_downstream_performance.csv")
    print("  - table6_model_scaling.csv")


if __name__ == "__main__":
    main()

