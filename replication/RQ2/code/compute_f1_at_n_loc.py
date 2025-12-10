#!/usr/bin/env python
"""
Compute F1@N%LOC for multiple N values.

F1@N%LOC metrics:
- K = ceil(N/100 * |U|) where U is all lines in the repo
- Take top-K lines as "reviewed"
- TP@K = |(top-K) ∩ P|
- FP@K = K - TP@K
- FN@K = |P| - TP@K
- Precision@K = TP@K / K
- Recall@K = TP@K / |P|
- F1@K = 2 * P@K * R@K / (P@K + R@K)
"""
import argparse
import math
from pathlib import Path

import pandas as pd
import yaml


def compute_f1_at_n_loc(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(config_path).parent.parent
    input_dir = base / "outputs" / "tables" / "intermediate"
    output_dir = base / "outputs" / "tables" / "intermediate"
    oracle_root = base / cfg["paths"]["oracle_root"]

    techs = cfg.get("techs", ["ansible", "chef", "puppet"])
    n_percentages = cfg.get("f1_n_percent", [0.5, 1, 2, 5, 10])

    # Load ground truth counts (constant per tech)
    gt_counts = {}
    for tech in techs:
        oracle = pd.read_csv(oracle_root / f"{tech}.csv")
        gt_counts[tech] = len(oracle)

    results = []

    for tech in techs:
        evalframe = pd.read_parquet(input_dir / f"{tech}__evalframe.parquet")
        total_positives_gt = gt_counts[tech]

        for method in evalframe["method"].unique():
            method_df = evalframe[evalframe["method"] == method].reset_index(drop=True)
            total_lines = len(method_df)  # |U| - all lines in repo
            method_tp_coverage = method_df["is_tp"].sum()  # TPs this method can reach (its coverage)

            for n_pct in n_percentages:
                k = math.ceil((n_pct / 100.0) * total_lines)
                k = min(k, total_lines)

                top_k = method_df.head(k)
                tp_at_k = top_k["is_tp"].sum()
                fp_at_k = k - tp_at_k
                fn_at_k = total_positives_gt - tp_at_k

                precision_at_k = tp_at_k / k if k > 0 else 0.0
                recall_at_k = tp_at_k / total_positives_gt if total_positives_gt > 0 else 0.0
                
                if precision_at_k + recall_at_k > 0:
                    f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
                else:
                    f1_at_k = 0.0

                results.append({
                    "tech": tech,
                    "method": method,
                    "n_percent_loc": n_pct,
                    "k_lines": k,
                    "total_lines": total_lines,
                    "total_positives_gt": total_positives_gt,
                    "method_tp_coverage": method_tp_coverage,
                    "tp_at_k": tp_at_k,
                    "fp_at_k": fp_at_k,
                    "fn_at_k": fn_at_k,
                    "precision_at_k": precision_at_k,
                    "recall_at_k": recall_at_k,
                    "f1_at_k": f1_at_k,
                })

        out_path = output_dir / f"{tech}__f1_at_n_loc.csv"
        tech_results = [r for r in results if r["tech"] == tech]
        pd.DataFrame(tech_results).to_csv(out_path, index=False)
        print(f"✅ {out_path.name}: {len(tech_results)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    compute_f1_at_n_loc(args.config)



