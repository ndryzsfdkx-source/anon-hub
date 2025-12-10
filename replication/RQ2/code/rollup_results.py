#!/usr/bin/env python
"""
Join all metrics into final tech-level and all-techs summary tables.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def rollup(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(config_path).parent.parent
    input_dir = base / "outputs" / "tables" / "intermediate"
    output_dir = base / "outputs" / "tables" / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    techs = cfg.get("techs", ["ansible", "chef", "puppet"])

    all_tech_frames = []

    for tech in techs:
        f1_data = pd.read_csv(input_dir / f"{tech}__f1_at_n_loc.csv")
        effort_data = pd.read_csv(input_dir / f"{tech}__effort_at_n_recall.csv")

        # Create wide-format F1 table
        f1_pivot = f1_data.pivot_table(
            index=["tech", "method", "total_lines", "total_positives_gt", "method_tp_coverage"],
            columns="n_percent_loc",
            values=["f1_at_k", "precision_at_k", "recall_at_k"],
            aggfunc="first"
        ).reset_index()
        f1_pivot.columns = [
            f"{col[0]}_at_{col[1]}pct" if col[1] != "" else col[0]
            for col in f1_pivot.columns
        ]

        # Create wide-format Effort table
        effort_pivot = effort_data.pivot_table(
            index=["tech", "method"],
            columns="target_recall_percent",
            values=["effort_percent", "reached"],
            aggfunc="first"
        ).reset_index()
        effort_pivot.columns = [
            f"{col[0]}_at_{col[1]}pct" if col[1] != "" else col[0]
            for col in effort_pivot.columns
        ]

        # Merge F1 and Effort
        merged = f1_pivot.merge(effort_pivot, on=["tech", "method"], how="left")

        out_path = output_dir / f"{tech}.csv"
        merged.to_csv(out_path, index=False)
        print(f"✅ {out_path.name}: {len(merged)} rows")

        all_tech_frames.append(merged)

    all_techs = pd.concat(all_tech_frames, ignore_index=True)
    all_path = output_dir / "all_techs.csv"
    all_techs.to_csv(all_path, index=False)
    print(f"✅ {all_path.name}: {len(all_techs)} rows across {len(techs)} techs")

    # Also create a paper-ready main results table
    paper_results = []
    for tech in techs:
        f1_data = pd.read_csv(input_dir / f"{tech}__f1_at_n_loc.csv")
        effort_data = pd.read_csv(input_dir / f"{tech}__effort_at_n_recall.csv")

        for method in f1_data["method"].unique():
            method_f1 = f1_data[f1_data["method"] == method]
            method_effort = effort_data[effort_data["method"] == method]

            row = {"tech": tech, "method": method}
            
            # Add F1 at key percentages
            for n_pct in [0.5, 1, 2, 5]:
                f1_row = method_f1[method_f1["n_percent_loc"] == n_pct]
                if len(f1_row) > 0:
                    pct_str = str(n_pct).replace('.', 'p')
                    row[f"f1_at_{pct_str}pct"] = f1_row.iloc[0]["f1_at_k"]
            
            # Add method coverage info
            if len(method_f1) > 0:
                row["total_positives_gt"] = method_f1.iloc[0]["total_positives_gt"]
                row["method_tp_coverage"] = method_f1.iloc[0]["method_tp_coverage"]
                row["max_recall"] = method_f1.iloc[0]["method_tp_coverage"] / method_f1.iloc[0]["total_positives_gt"]

            # Add Effort at key recalls
            for recall_pct in [60, 80, 90]:
                effort_row = method_effort[method_effort["target_recall_percent"] == recall_pct]
                if len(effort_row) > 0:
                    row[f"effort_at_{recall_pct}pct_recall"] = effort_row.iloc[0]["effort_percent"]
                    row[f"reached_{recall_pct}pct"] = effort_row.iloc[0]["reached"]

            paper_results.append(row)

    paper_df = pd.DataFrame(paper_results)
    paper_path = output_dir / "paper_main_results.csv"
    paper_df.to_csv(paper_path, index=False)
    print(f"✅ {paper_path.name}: {len(paper_df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    rollup(args.config)


