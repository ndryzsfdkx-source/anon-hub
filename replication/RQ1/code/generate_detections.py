#!/usr/bin/env python3
"""Generate standardized detection CSVs for all methods.

This script aggregates detections from:
  * Oracle labels (ground truth)
  * Static analysis tools (GLITCH, SLAC/SLIC)
  * IntelliSA approach (intellisa/testset-*.csv)
  * LLM providers (responses/*)

Outputs are written under the detections/ directory, partitioned by method.
Each CSV uses the unified schema: PATH,LINE,CATEGORY with canonicalized smell names.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

from evaluate import (
    TECHS,
    Event,
    EventSet,
    display_name_for_method,
    load_glitch_detections,
    load_llm_predictions,
    load_oracle_labels,
    load_intellisa_detections,
    load_slac_slic_detections,
    normalize_category,
)

# Prefer the first existing candidate under the given root; otherwise return the first candidate.
def resolve_dir(root: Path, path: Path, fallbacks: Sequence[str]) -> Path:
    def expand(p: Path) -> Path:
        return p if p.is_absolute() else root / p

    primary = expand(path)
    if primary.exists():
        return primary

    for rel in fallbacks:
        candidate = root / rel
        if candidate.exists():
            return candidate

    return primary


@dataclass
class MethodDetections:
    name: str
    events_by_tech: Mapping[str, Optional[EventSet]]
    metadata: Mapping[str, object]


# Header shared by all exported CSV files
CSV_HEADER = ("PATH", "LINE", "CATEGORY")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate standardized detection CSVs across all methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing data/output (default: current directory).",
    )
    parser.add_argument("--detections", type=Path, default=Path("detections"), help="Output directory root")
    parser.add_argument("--labels", type=Path, default=Path("label"), help="Directory with oracle CSVs")
    parser.add_argument("--glitch", type=Path, default=Path("glitch"), help="Directory with GLITCH detections")
    parser.add_argument("--slac-slic", dest="slac_slic", type=Path, default=Path("SLAC_SLIC"), help="Directory with SLAC/SLIC detections")
    parser.add_argument("--intellisa", type=Path, default=Path("intellisa"), help="Directory with IntelliSA detections")
    parser.add_argument("--responses", type=Path, default=Path("responses"), help="Directory with LLM response JSON files")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing detections directory before generating new files.",
    )
    return parser


def sanitize_method_key(method_key: str) -> str:
    token = method_key.replace("::", "__")
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    token = token.strip("_")
    return token or "method"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_metadata(method_dir: Path, payload: Mapping[str, object]) -> None:
    meta_path = method_dir / "_meta.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalized_events(events: Optional[Iterable[Event]]) -> Sequence[Event]:
    if not events:
        return []
    cleaned: Set[Event] = set()
    for path_value, line_no, category in events:
        if line_no is None or line_no < 0:
            continue
        normalized_category = normalize_category(category)
        if not normalized_category:
            continue
        cleaned.add((path_value, line_no, normalized_category))
    return sorted(cleaned, key=lambda item: (item[0], item[1], item[2]))


def write_detection_csv(path: Path, events: Optional[Iterable[Event]]) -> None:
    ensure_dir(path.parent)
    rows = normalized_events(events)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for path_value, line_no, category in rows:
            writer.writerow([path_value, line_no, category])


def build_static_methods(
    labels_dir: Path,
    glitch_dir: Path,
    slac_slic_dir: Path,
    intellisa_dir: Path,
) -> Sequence[MethodDetections]:
    methods = []

    oracle = load_oracle_labels(labels_dir)
    methods.append(
        MethodDetections(
            name="oracle",
            events_by_tech=oracle,
            metadata={"source": "oracle"},
        )
    )

    glitch = load_glitch_detections(glitch_dir)
    methods.append(
        MethodDetections(
            name="glitch",
            events_by_tech=glitch,
            metadata={"source": "static", "tool": "GLITCH"},
        )
    )

    slac_slic = load_slac_slic_detections(slac_slic_dir)
    methods.append(
        MethodDetections(
            name="slac_slic",
            events_by_tech=slac_slic,
            metadata={"source": "static", "tool": "SLAC/SLIC"},
        )
    )

    intellisa = load_intellisa_detections(intellisa_dir)
    methods.append(
        MethodDetections(
            name="intellisa",
            events_by_tech=intellisa,
            metadata={"source": "custom"},
        )
    )

    return methods


def build_llm_methods(responses_dir: Path) -> Sequence[MethodDetections]:
    systems = load_llm_predictions(responses_dir)
    methods: list[MethodDetections] = []
    for method_key, events_by_tech in sorted(systems.items()):
        method_slug = sanitize_method_key(method_key)
        provider = None
        model = None
        style = None
        parts = method_key.split("::")
        if len(parts) == 3:
            provider, model, style = parts
        elif len(parts) == 2:
            provider, model = parts
        metadata: Dict[str, object] = {
            "source": "llm",
            "method_key": method_key,
            "display_name": display_name_for_method(method_key),
        }
        if provider:
            metadata["provider"] = provider
        if model:
            metadata["model"] = model
        if style:
            metadata["style"] = style
        methods.append(
            MethodDetections(
                name=method_slug,
                events_by_tech=events_by_tech,
                metadata=metadata,
            )
        )
    return methods


def write_method_detections(output_root: Path, method: MethodDetections) -> None:
    method_dir = output_root / method.name
    ensure_dir(method_dir)
    write_metadata(method_dir, method.metadata)

    tech_keys = set(method.events_by_tech.keys()) | set(TECHS)
    for tech in sorted(tech_keys):
        events = method.events_by_tech.get(tech)
        path = method_dir / f"{tech}.csv"
        write_detection_csv(path, events)


def main() -> None:
    args = build_parser().parse_args()

    root = args.root.resolve()

    output_root = resolve_dir(root, args.detections, ["data/detections"])
    labels_dir = resolve_dir(root, args.labels, ["data/oracle"])
    glitch_dir = resolve_dir(root, args.glitch, ["data/baselines/glitch"])
    slac_slic_dir = resolve_dir(root, args.slac_slic, ["data/baselines/SLAC_SLIC"])
    intellisa_dir = resolve_dir(root, args.intellisa, ["data/intellisa"])
    responses_dir = resolve_dir(root, args.responses, ["data/responses"])

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root)

    static_methods = build_static_methods(labels_dir, glitch_dir, slac_slic_dir, intellisa_dir)
    llm_methods = build_llm_methods(responses_dir)

    all_methods = static_methods + llm_methods
    if not all_methods:
        print("No detections found to export.")
        return

    for method in all_methods:
        write_method_detections(output_root, method)
        print(f"Wrote detections for {method.metadata.get('display_name', method.name)} -> {output_root / method.name}")


if __name__ == "__main__":
    main()
