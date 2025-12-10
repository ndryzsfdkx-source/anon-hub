#!/usr/bin/env python
"""
Generate basic statistics table showing method coverage and alerts.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def generate_statistics(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(config_path).parent.parent
    input_dir = base / "outputs" / "tables" / "intermediate"
    ranking_root = base / cfg["paths"]["ranking_root"]
    output_dir = base / "outputs" / "tables" / "final"

    techs = cfg.get("techs", ["ansible", "chef", "puppet"])

    print("=" * 80)
    print("GENERATING STATISTICS TABLE")
    print("=" * 80)

    stats = []

    for tech in techs:
        print(f"\nProcessing {tech}...")
        
        # Get basic info from F1 data
        f1_df = pd.read_csv(input_dir / f"{tech}__f1_at_n_loc.csv")
        
        for method in f1_df["method"].unique():
            method_data = f1_df[f1_df["method"] == method].iloc[0]
            
            # Count alerts (lines with score > 0) from ranking file
            ranking_file = ranking_root / tech / f"{method}.csv"
            if ranking_file.exists():
                ranking_df = pd.read_csv(ranking_file)
                # Alerts are lines with non-zero scores
                method_alerts = (ranking_df["score"] > 0).sum()
            else:
                method_alerts = None
            
            stats.append({
                "tech": tech,
                "method": method,
                "total_lines": int(method_data["total_lines"]),
                "total_positives_gt": int(method_data["total_positives_gt"]),
                "method_tp_coverage": int(method_data["method_tp_coverage"]),
                "method_alerts": method_alerts,
                "max_recall": method_data["method_tp_coverage"] / method_data["total_positives_gt"],
                "alert_density": method_alerts / method_data["total_lines"] if method_alerts else None,
                "tp_precision": method_data["method_tp_coverage"] / method_alerts if method_alerts and method_alerts > 0 else None
            })

    stats_df = pd.DataFrame(stats)
    
    # Display and save
    print("\n" + "=" * 80)
    print("STATISTICS TABLE")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    
    stats_df.to_csv(output_dir / "method_statistics.csv", index=False)
    print(f"\n✅ Saved: {output_dir / 'method_statistics.csv'}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY BY TECH")
    print("=" * 80)
    for tech in techs:
        tech_stats = stats_df[stats_df["tech"] == tech]
        print(f"\n{tech.upper()}:")
        print(f"  Total lines: {tech_stats.iloc[0]['total_lines']:,}")
        print(f"  GT positives: {tech_stats.iloc[0]['total_positives_gt']}")
        print(f"  GT density: {tech_stats.iloc[0]['total_positives_gt']/tech_stats.iloc[0]['total_lines']:.2%}")
        print(f"  Method coverage range: {tech_stats['method_tp_coverage'].min()}-{tech_stats['method_tp_coverage'].max()} TPs")
        print(f"  Method alerts range: {tech_stats['method_alerts'].min():.0f}-{tech_stats['method_alerts'].max():.0f} alerts")
        print(f"  Avg TP precision: {tech_stats['tp_precision'].mean():.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    generate_statistics(args.config)

