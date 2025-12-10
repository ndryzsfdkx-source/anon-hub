#!/usr/bin/env python
"""
Generate rankings for F1@N%LOC and Effort@N%Recall across all configurations.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def generate_rankings(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(config_path).parent.parent
    input_dir = base / "outputs" / "tables" / "intermediate"
    output_dir = base / "outputs" / "tables" / "final"

    techs = cfg.get("techs", ["ansible", "chef", "puppet"])
    n_percentages = cfg.get("f1_n_percent", [0.5, 1, 2, 5, 10])
    recall_targets = cfg.get("effort_recall_targets", [60, 80, 90])

    # ===== F1@N%LOC Rankings =====
    print("=" * 80)
    print("F1@N%LOC RANKINGS")
    print("=" * 80)
    
    f1_rankings = []
    
    for n_pct in n_percentages:
        print(f"\n📊 F1@{n_pct}%LOC Rankings:")
        print("-" * 80)
        
        # Per-tech rankings
        for tech in techs:
            df = pd.read_csv(input_dir / f"{tech}__f1_at_n_loc.csv")
            df_n = df[df["n_percent_loc"] == n_pct].copy()
            df_n = df_n.sort_values("f1_at_k", ascending=False).reset_index(drop=True)
            df_n["rank"] = range(1, len(df_n) + 1)
            
            print(f"\n{tech.upper()}:")
            for _, row in df_n.iterrows():
                print(f"  {row['rank']}. {row['method']}: {row['f1_at_k']:.4f}")
                f1_rankings.append({
                    "metric": f"f1_at_{n_pct}pct",
                    "tech": tech,
                    "method": row["method"],
                    "value": row["f1_at_k"],
                    "rank": row["rank"]
                })
        
        # Combined ranking (average F1 across techs)
        all_data = []
        for tech in techs:
            df = pd.read_csv(input_dir / f"{tech}__f1_at_n_loc.csv")
            df_n = df[df["n_percent_loc"] == n_pct][["method", "f1_at_k"]]
            all_data.append(df_n)
        
        combined = pd.concat(all_data).groupby("method")["f1_at_k"].mean().reset_index()
        combined = combined.sort_values("f1_at_k", ascending=False).reset_index(drop=True)
        combined["rank"] = range(1, len(combined) + 1)
        
        print(f"\nCOMBINED (avg across 3 techs):")
        for _, row in combined.iterrows():
            print(f"  {row['rank']}. {row['method']}: {row['f1_at_k']:.4f}")
            f1_rankings.append({
                "metric": f"f1_at_{n_pct}pct",
                "tech": "combined",
                "method": row["method"],
                "value": row["f1_at_k"],
                "rank": row["rank"]
            })
    
    # Save F1 rankings
    f1_df = pd.DataFrame(f1_rankings)
    f1_df.to_csv(output_dir / "f1_rankings.csv", index=False)
    print(f"\n✅ Saved: {output_dir / 'f1_rankings.csv'}")

    # ===== Effort@N%Recall Rankings =====
    print("\n" + "=" * 80)
    print("EFFORT@N%RECALL RANKINGS")
    print("=" * 80)
    
    effort_rankings = []
    
    for recall_pct in recall_targets:
        print(f"\n💪 Effort@{recall_pct}%Recall Rankings:")
        print("-" * 80)
        
        # Per-tech rankings
        for tech in techs:
            df = pd.read_csv(input_dir / f"{tech}__effort_at_n_recall.csv")
            df_n = df[df["target_recall_percent"] == recall_pct].copy()
            
            # Sort: reached first (by effort ascending), then unreached (by max_recall descending)
            df_n["sort_key1"] = df_n["reached"].map({True: 0, False: 1})
            df_n["sort_key2"] = df_n.apply(
                lambda x: x["effort_percent"] if x["reached"] else -x["max_recall"],
                axis=1
            )
            df_n = df_n.sort_values(["sort_key1", "sort_key2"]).reset_index(drop=True)
            df_n["rank"] = range(1, len(df_n) + 1)
            
            print(f"\n{tech.upper()}:")
            for _, row in df_n.iterrows():
                if row["reached"]:
                    print(f"  {row['rank']}. {row['method']}: {row['effort_percent']:.2f}% ✓")
                else:
                    print(f"  {row['rank']}. {row['method']}: unreached (max={row['max_recall']:.1%})")
                
                effort_rankings.append({
                    "metric": f"effort_at_{recall_pct}pct",
                    "tech": tech,
                    "method": row["method"],
                    "effort_percent": row["effort_percent"],
                    "reached": row["reached"],
                    "max_recall": row["max_recall"],
                    "rank": row["rank"]
                })
        
        # Combined ranking (average effort for reached methods)
        all_data = []
        for tech in techs:
            df = pd.read_csv(input_dir / f"{tech}__effort_at_n_recall.csv")
            df_n = df[df["target_recall_percent"] == recall_pct][
                ["method", "effort_percent", "reached", "max_recall"]
            ]
            all_data.append(df_n)
        
        combined = pd.concat(all_data).groupby("method").agg({
            "effort_percent": "mean",
            "reached": lambda x: x.sum() > 0,  # reached in at least 1 tech
            "max_recall": "mean"
        }).reset_index()
        
        # Sort
        combined["sort_key1"] = combined["reached"].map({True: 0, False: 1})
        combined["sort_key2"] = combined.apply(
            lambda x: x["effort_percent"] if x["reached"] else -x["max_recall"],
            axis=1
        )
        combined = combined.sort_values(["sort_key1", "sort_key2"]).reset_index(drop=True)
        combined["rank"] = range(1, len(combined) + 1)
        
        print(f"\nCOMBINED (avg across 3 techs):")
        for _, row in combined.iterrows():
            if row["reached"]:
                print(f"  {row['rank']}. {row['method']}: {row['effort_percent']:.2f}% ✓")
            else:
                print(f"  {row['rank']}. {row['method']}: mostly unreached (avg max={row['max_recall']:.1%})")
            
            effort_rankings.append({
                "metric": f"effort_at_{recall_pct}pct",
                "tech": "combined",
                "method": row["method"],
                "effort_percent": row["effort_percent"],
                "reached": row["reached"],
                "max_recall": row["max_recall"],
                "rank": row["rank"]
            })
    
    # Save Effort rankings
    effort_df = pd.DataFrame(effort_rankings)
    effort_df.to_csv(output_dir / "effort_rankings.csv", index=False)
    print(f"\n✅ Saved: {output_dir / 'effort_rankings.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    generate_rankings(args.config)

