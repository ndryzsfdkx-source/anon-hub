#!/usr/bin/env python
"""
Compute Effort@N%Recall for multiple recall target values.

Effort@N%Recall metrics:
- Given a target recall r (e.g., 0.6, 0.8, 0.9)
- Find the smallest K such that Recall@K >= r
- Report Effort% = 100 * K / |U|
- If r is unreachable (method's max recall < r), return 100% and mark as unreached
"""
import argparse
import math
from pathlib import Path

import pandas as pd
import yaml


def compute_effort_at_n_recall(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(config_path).parent.parent
    input_dir = base / "outputs" / "tables" / "intermediate"
    output_dir = base / "outputs" / "tables" / "intermediate"
    oracle_root = base / cfg["paths"]["oracle_root"]

    techs = cfg.get("techs", ["ansible", "chef", "puppet"])
    recall_targets = cfg.get("effort_recall_targets", [60, 80, 90])

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
            method_tp_coverage = method_df["is_tp"].sum()  # TPs this method can reach

            if method_tp_coverage == 0:
                # Edge case: no positives
                for target_pct in recall_targets:
                    results.append({
                        "tech": tech,
                        "method": method,
                        "target_recall_percent": target_pct,
                        "effort_percent": 100.0,
                        "k_lines": total_lines,
                        "total_lines": total_lines,
                        "total_positives_gt": total_positives_gt,
                        "method_tp_coverage": 0,
                        "max_recall": 0.0,
                        "reached": False,
                    })
                continue

            # Compute cumulative recall (relative to GT)
            method_df["cumulative_tp"] = method_df["is_tp"].cumsum()
            method_df["cumulative_recall"] = method_df["cumulative_tp"] / total_positives_gt
            
            # Maximum achievable recall (method's coverage relative to GT)
            max_recall = method_tp_coverage / total_positives_gt

            for target_pct in recall_targets:
                target_recall = target_pct / 100.0

                # Check if target is reachable
                if max_recall < target_recall:
                    results.append({
                        "tech": tech,
                        "method": method,
                        "target_recall_percent": target_pct,
                        "effort_percent": 100.0,
                        "k_lines": total_lines,
                        "total_lines": total_lines,
                        "total_positives_gt": total_positives_gt,
                        "method_tp_coverage": method_tp_coverage,
                        "max_recall": max_recall,
                        "reached": False,
                    })
                    continue

                # Find smallest K where recall >= target
                reached_rows = method_df[method_df["cumulative_recall"] >= target_recall]
                
                if len(reached_rows) == 0:
                    # Unreachable
                    k_lines = total_lines
                    effort_percent = 100.0
                    reached = False
                else:
                    k_lines = reached_rows.index[0] + 1
                    effort_percent = 100.0 * k_lines / total_lines
                    reached = True

                results.append({
                    "tech": tech,
                    "method": method,
                    "target_recall_percent": target_pct,
                    "effort_percent": effort_percent,
                    "k_lines": k_lines,
                    "total_lines": total_lines,
                    "total_positives_gt": total_positives_gt,
                    "method_tp_coverage": method_tp_coverage,
                    "max_recall": max_recall,
                    "reached": reached,
                })

        out_path = output_dir / f"{tech}__effort_at_n_recall.csv"
        tech_results = [r for r in results if r["tech"] == tech]
        pd.DataFrame(tech_results).to_csv(out_path, index=False)
        print(f"✅ {out_path.name}: {len(tech_results)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    compute_effort_at_n_recall(args.config)



