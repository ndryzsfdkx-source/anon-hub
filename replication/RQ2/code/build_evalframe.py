#!/usr/bin/env python
"""
Join ranked lists with oracle to create evaluation frames per tech.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def create_evalframe(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(config_path).parent.parent
    ranking_root = base / cfg["paths"]["ranking_root"]
    oracle_root = base / cfg["paths"]["oracle_root"]
    output_dir = base / "outputs" / "tables" / "intermediate"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_output_dir = base / "outputs" / "ranking_with_label"
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    techs = cfg.get("techs", ["ansible", "chef", "puppet"])

    for tech in techs:
        print(f"Processing {tech}...")

        oracle = pd.read_csv(oracle_root / f"{tech}.csv")
        oracle.columns = oracle.columns.str.lower()
        oracle.rename(columns={"path": "file_path", "category": "smell"}, inplace=True)
        oracle["is_tp"] = 1

        tech_dir = ranking_root / tech
        ranking_csvs = sorted(tech_dir.glob("*.csv"))

        frames = []
        for csv_path in ranking_csvs:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.lower()

            merged = df.merge(
                oracle[["file_path", "line", "smell", "is_tp"]],
                on=["file_path", "line", "smell"],
                how="left"
            )
            merged["is_tp"] = merged["is_tp"].fillna(0).astype(int)
            frames.append(merged)

        evalframe = pd.concat(frames, ignore_index=True)
        evalframe = evalframe.sort_values(["method", "rank"]).reset_index(drop=True)

        out_path = output_dir / f"{tech}__evalframe.parquet"
        evalframe.to_parquet(out_path, index=False)
        print(f"  ✅ {out_path.name}: {len(evalframe)} rows, {evalframe['method'].nunique()} methods")
        
        csv_path = csv_output_dir / f"{tech}__ranking_with_label.csv"
        evalframe.to_csv(csv_path, index=False)
        print(f"  ✅ {csv_path.name}: {len(evalframe)} rows (CSV for human reading)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    create_evalframe(args.config)


