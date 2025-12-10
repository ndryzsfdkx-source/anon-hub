#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def resolve_oracle_dir(root: Path) -> Path:
    for rel in ("data/oracle", "label", "oracle"):
        candidate = root / rel
        if candidate.exists():
            return candidate
    return root / "data/oracle"


def analyze_oracle(filename: Path, tech_name: str):
    print(f"\n{'='*60}")
    print(f"Analyzing {tech_name} Oracle Dataset")
    print('='*60)
    
    with filename.open('r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Group by script path
    scripts = defaultdict(list)
    for row in rows:
        scripts[row['PATH']].append(row)
    
    total_scripts = len(scripts)
    smell_free_scripts = 0
    scripts_with_smells = 0
    total_smell_instances = 0
    
    for path, entries in scripts.items():
        # Check if script is smell-free
        is_smell_free = all(
            entry['LINE'] == '0' and entry['CATEGORY'] == 'none' 
            for entry in entries
        )
        
        if is_smell_free:
            smell_free_scripts += 1
        else:
            scripts_with_smells += 1
            # Count smell instances (entries where CATEGORY != 'none')
            smells_in_script = [e for e in entries if e['CATEGORY'] != 'none']
            total_smell_instances += len(smells_in_script)
    
    print(f"Total scripts: {total_scripts}")
    print(f"Scripts with smells: {scripts_with_smells}")
    print(f"Smell-free scripts: {smell_free_scripts}")
    print(f"Total smell instances: {total_smell_instances}")
    
    return {
        'total_scripts': total_scripts,
        'scripts_with_smells': scripts_with_smells,
        'smell_free_scripts': smell_free_scripts,
        'total_smell_instances': total_smell_instances
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize oracle datasets for Ansible/Chef/Puppet.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing data/oracle (default: current directory).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    oracle_dir = resolve_oracle_dir(root)

    ansible_stats = analyze_oracle(oracle_dir / 'oracle-dataset-ansible.csv', 'Ansible')
    chef_stats = analyze_oracle(oracle_dir / 'oracle-dataset-chef.csv', 'Chef')
    puppet_stats = analyze_oracle(oracle_dir / 'oracle-dataset-puppet.csv', 'Puppet')

    # Summary for LaTeX text
    print(f"\n{'='*60}")
    print("SUMMARY FOR LATEX")
    print('='*60)
    print(f"\nAnsible:")
    print(f"  - Total scripts: {ansible_stats['total_scripts']}")
    print(f"  - Smell instances: {ansible_stats['total_smell_instances']}")
    print(f"  - Smell-free scripts: {ansible_stats['smell_free_scripts']}")

    print(f"\nChef:")
    print(f"  - Total scripts: {chef_stats['total_scripts']}")
    print(f"  - Smell instances: {chef_stats['total_smell_instances']}")
    print(f"  - Smell-free scripts: {chef_stats['smell_free_scripts']}")

    print(f"\nPuppet:")
    print(f"  - Total scripts: {puppet_stats['total_scripts']}")
    print(f"  - Smell instances: {puppet_stats['total_smell_instances']}")
    print(f"  - Smell-free scripts: {puppet_stats['smell_free_scripts']}")


if __name__ == "__main__":
    main()
