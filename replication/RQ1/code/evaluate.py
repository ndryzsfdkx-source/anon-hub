import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

TECHS = ("ansible", "chef", "puppet")
CATEGORY_ALIASES = {
    "admin by default": "Admin by default",
    "admin-by-default": "Admin by default",
    "empty password": "Empty password",
    "empty-password": "Empty password",
    "hard coded secret": "Hard-coded secret",
    "hard-coded secret": "Hard-coded secret",
    "hardcoded secret": "Hard-coded secret",
    "hardcoded-password": "Hard-coded secret",
    "hardcoded password": "Hard-coded secret",
    "hardcoded-username": "Hard-coded secret",
    "missing default in case statement": "Missing Default in Case Statement",
    "missing default case statement": "Missing Default in Case Statement",
    "missing default switch": "Missing Default in Case Statement",
    "missing default in switch statement": "Missing Default in Case Statement",
    "missing default in case statements": "Missing Default in Case Statement",
    "no integrity check": "No integrity check",
    "no-integrity-check": "No integrity check",
    "suspicious comment": "Suspicious comment",
    "unrestricted ip address": "Unrestricted IP Address",
    "improper ip address binding": "Unrestricted IP Address",
    "use of http": "Use of HTTP without SSL/TLS",
    "use of http without ssl/tls": "Use of HTTP without SSL/TLS",
    "http without ssl/tls": "Use of HTTP without SSL/TLS",
    "use of weak cryptography algorithms": "Use of weak cryptography algorithms",
    "weak cryptography algorithms": "Use of weak cryptography algorithms",
    "weak cryptography algorithm": "Use of weak cryptography algorithms",
    "weak cryptography": "Use of weak cryptography algorithms",
    "none": None,
    "no issues found": None,
    "no issue": None,
    "no-issue": None,
    "no issues": None,
    "n/a": None,
}
LINE_PATTERN = re.compile(r"-?\d+")
PATH_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_./@-]+$")
PLACEHOLDER_PATH_TOKENS = {
    "NAME_OF_FILE",
    "FILE_NAME",
    "FILENAME",
    "TARGET_FILE",
    "PATH_TO_FILE",
}

# Prefer the first existing candidate under the given root; otherwise return the first candidate.
def resolve_dir(root: Path, candidates: List[str]) -> Path:
    for rel in candidates:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return root / candidates[0]


# Canonical ordering of the 9 security smells
SMELL_ORDER = [
    # Requested display order for per-smell table
    "Admin by default",
    "Empty password",
    "Hard-coded secret",
    "Unrestricted IP Address",  # displayed as Invalid/Improper IP address binding
    "Suspicious comment",
    "Use of HTTP without SSL/TLS",  # displayed as Use of HTTP without TLS
    "No integrity check",
    "Use of weak cryptography algorithms",  # displayed as Use of weak crypto alg.
    "Missing Default in Case Statement",  # displayed as Missing default case statement
]

# Short labels for console columns (keep the table readable)
SMELL_SHORT = {
    "Admin by default": "Admin",
    "Empty password": "EmptyPwd",
    "Hard-coded secret": "HardSecret",
    "Missing Default in Case Statement": "MissingDefault",
    "No integrity check": "NoIntegrity",
    "Suspicious comment": "Suspicious",
    "Unrestricted IP Address": "InvalidIP",
    "Use of HTTP without SSL/TLS": "HTTPnoTLS",
    "Use of weak cryptography algorithms": "WeakCrypto",
}

# Full display labels for per-smell table headers (only affects console header, not CSV values)
SMELL_DISPLAY = {
    "Admin by default": "Admin by default",
    "Empty password": "Empty password",
    "Hard-coded secret": "Hard-coded secret",
    "Unrestricted IP Address": "Invalid IP address binding",
    "Suspicious comment": "Suspicious comment",
    "Use of HTTP without SSL/TLS": "Use of HTTP without TLS",
    "No integrity check": "No integrity check",
    "Use of weak cryptography algorithms": "Use of weak crypto alg.",
    "Missing Default in Case Statement": "Missing default case statement",
}


Event = Tuple[str, int, str]
EventSet = Set[Event]
SystemPredictions = Dict[str, Dict[str, EventSet]]


def sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    cleaned = cleaned.strip("_")
    return cleaned or "model"


def normalize_path(value: str) -> str:
    return value.strip()


def normalize_category(value: str) -> Optional[str]:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    key = raw.lower()
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    key = key.replace("-", " ").replace("_", " ")
    key = re.sub(r"\s+", " ", key).strip()
    key = key.lower()
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    return raw


def parse_line_number(value: str) -> Optional[int]:
    match = LINE_PATTERN.search(value)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def parse_detection_lines(text: str) -> List[Event]:
    events: List[Event] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("\u2028\u2029")
        if not line:
            continue
        if line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",", 2)]
        if len(parts) < 3:
            continue
        path_part, line_part, category_part = parts[0], parts[1], parts[2]
        line_no = parse_line_number(line_part)
        if line_no is None:
            continue
        category = normalize_category(category_part)
        if not category:
            continue
        path = normalize_path(path_part)
        if not path:
            continue
        if not PATH_TOKEN_PATTERN.match(path):
            continue
        events.append((path, line_no, category))
    return events


def extract_prompt_target(stem: str, style: Optional[str]) -> Optional[str]:
    prefix = f"prompt_{style}_" if style else "prompt_"
    if stem.startswith(prefix):
        candidate = stem[len(prefix) :]
    elif stem.startswith("prompt_"):
        candidate = stem[len("prompt_") :]
    else:
        candidate = stem
    if "_" in candidate:
        candidate = candidate.rsplit("_", 1)[0]
    candidate = candidate.strip()
    return candidate or None


def replace_placeholder_paths(events: List[Event], fallback: Optional[str]) -> List[Event]:
    if not events or not fallback:
        return events
    normalized_fallback = normalize_path(fallback)
    return [
        (normalized_fallback if path.upper() in PLACEHOLDER_PATH_TOKENS else path, line_no, category)
        for path, line_no, category in events
    ]


def detect_style_from_filename(name: str) -> Optional[str]:
    if name.startswith("prompt_definition_based_"):
        return "definition_based"
    if name.startswith("prompt_static_analysis_rules_"):
        return "static_analysis_rules"
    return None


def load_csv_events(path: Path, path_field: str, line_field: str, category_field: str) -> EventSet:
    events: EventSet = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = normalize_category(row.get(category_field, ""))
            if not category:
                continue
            line_no = parse_line_number(str(row.get(line_field, "")))
            if line_no is None:
                continue
            norm_path = normalize_path(row.get(path_field, ""))
            if not norm_path:
                continue
            events.add((norm_path, line_no, category))
    return events


def load_oracle_labels(label_dir: Path) -> Dict[str, EventSet]:
    result: Dict[str, EventSet] = {}
    for tech in TECHS:
        path = label_dir / f"oracle-dataset-{tech}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing oracle labels for {tech}: {path}")
        result[tech] = load_csv_events(path, "PATH", "LINE", "CATEGORY")
    return result


def load_glitch_detections(glitch_dir: Path) -> Dict[str, EventSet]:
    result: Dict[str, EventSet] = {}
    for tech in TECHS:
        path = glitch_dir / f"GLITCH-{tech}-oracle.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing GLITCH detections for {tech}: {path}")
        result[tech] = load_csv_events(path, "PATH", "LINE", "ERROR")
    return result


def load_slac_slic_detections(slac_slic_dir: Path) -> Dict[str, Optional[EventSet]]:
    mapping = {
        "ansible": slac_slic_dir / "SLAC-ansible-oracle.csv",
        "chef": slac_slic_dir / "SLAC-chef-oracle.csv",
        "puppet": slac_slic_dir / "SLIC-puppet-oracle.csv",
    }
    result: Dict[str, Optional[EventSet]] = {}
    for tech, path in mapping.items():
        if not path.exists():
            result[tech] = None
            continue
        # SLAC/SLIC files use the same column naming as GLITCH (ERROR)
        result[tech] = load_csv_events(path, "PATH", "LINE", "ERROR")
    return result


def load_intellisa_detections(intellisa_dir: Path) -> Dict[str, EventSet]:
    mapping = {
        "ansible": intellisa_dir / "testset-ansible.csv",
        "chef": intellisa_dir / "testset-chef.csv",
        "puppet": intellisa_dir / "testset-puppet.csv",
    }
    result: Dict[str, EventSet] = {}
    for tech, path in mapping.items():
        if not path.exists():
            result[tech] = set()
            continue
        result[tech] = load_csv_events(path, "PATH", "LINE", "CATEGORY")
    return result


def extract_openai_predictions(data: dict) -> List[Event]:
    outputs = []
    if "output" in data:
        outputs.extend(data.get("output", []))
    elif "choices" in data:
        outputs.extend(choice.get("message", {}) for choice in data.get("choices", []))
    events: List[Event] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("output_text")
                    if isinstance(text, str):
                        events.extend(parse_detection_lines(text))
                elif isinstance(block, str):
                    events.extend(parse_detection_lines(block))
        elif isinstance(content, str):
            events.extend(parse_detection_lines(content))
        text = item.get("text")
        if isinstance(text, str):
            events.extend(parse_detection_lines(text))
    return events


def extract_anthropic_predictions(data: dict) -> List[Event]:
    events: List[Event] = []
    for block in data.get("content", []):
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                events.extend(parse_detection_lines(text))
        elif isinstance(block, str):
            events.extend(parse_detection_lines(block))
    return events


def extract_openrouter_predictions(data: dict) -> List[Event]:
    events: List[Event] = []
    for choice in data.get("choices", []):
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            events.extend(parse_detection_lines(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    events.extend(parse_detection_lines(part))
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str):
                        events.extend(parse_detection_lines(text))
    return events


def extract_xai_predictions(data: dict) -> List[Event]:
    text = data.get("content")
    if isinstance(text, str):
        return parse_detection_lines(text)
    events: List[Event] = []
    for value in data.values():
        if isinstance(value, str):
            events.extend(parse_detection_lines(value))
    return events


EXTRACTORS = {
    "openai": extract_openai_predictions,
    "anthropic": extract_anthropic_predictions,
    "openrouter": extract_openrouter_predictions,
    "xai": extract_xai_predictions,
}


FRIENDLY_MODEL_NAMES = {
    "claude-sonnet-4-20250514": "claude-sonnet-4.0",
    "anthropic/claude-sonnet-4": "claude-sonnet-4.0",
    "x-ai/grok-4-fast:free": "grok-4-fast",
    "x-ai/grok-4-fast-non-reasoning": "grok-4-fast-non-reasoning",
    "openai/gpt-5": "gpt-5",
}

DISPLAY_NAME_OVERRIDES = {
    "anthropic::claude-sonnet-4-20250514::static_analysis_rules": "claude-sonnet-4.0-static-analysis-rules",
    "openrouter::x-ai/grok-4-fast:free::definition_based": "grok-4-fast-definition-based",
    "openrouter::x-ai/grok-4-fast:free::static_analysis_rules": "grok-4-fast-static-analysis-rules",
    "openrouter::anthropic/claude-sonnet-4::static_analysis_rules": "claude-sonnet-4.0-static-analysis-rules (OpenRouter)",
    "openrouter::anthropic/claude-sonnet-4::definition_based": "claude-sonnet-4.0-definition-based (OpenRouter)",
    "openrouter::openai/gpt-5::definition_based": "gpt-5-definition-based",
    "openrouter::openai/gpt-5::static_analysis_rules": "gpt-5-static-analysis-rules",
}


def friendly_model_name(model: Optional[str]) -> str:
    if not model:
        return "unknown-model"
    return FRIENDLY_MODEL_NAMES.get(model, model)


def display_name_for_method(method_key: str) -> str:
    if method_key in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[method_key]
    parts = method_key.split("::")
    if len(parts) == 3:
        provider, model, style = parts
        model_name = friendly_model_name(model)
        return f"{model_name}-{style.replace('_', '-')}"
    if len(parts) == 2:
        provider, model = parts
        model_name = friendly_model_name(model)
        return f"{provider}-{model_name}"
    return method_key


def normalize_style_token(token: str) -> Optional[str]:
    cleaned = token.strip().lower().replace("-", "_")
    cleaned = re.sub(r"[^a-z_]+", "", cleaned)
    if not cleaned:
        return None
    mapping = {
        "definitionbased": "definition_based",
        "definition_based": "definition_based",
        "definition": "definition_based",
        "staticanalysisrules": "static_analysis_rules",
        "static_analysis_rules": "static_analysis_rules",
        "static": "static_analysis_rules",
    }
    return mapping.get(cleaned, cleaned)


def parse_model_spec(spec: str) -> Optional[str]:
    token = spec.strip()
    if "-" not in token:
        return None
    head, style_token = token.rsplit("-", 1)
    style = normalize_style_token(style_token)
    if style is None:
        return None
    if "/" in head:
        provider_part, model_part = head.split("/", 1)
    elif "-" in head:
        provider_part, model_part = head.split("-", 1)
    else:
        return None
    provider_clean = provider_part.strip().lower()
    model = model_part.strip()
    if not provider_clean or not model:
        return None
    return f"{provider_clean}::{model}::{style}"


BASELINE_SPECS = ["GLITCH", "SLAC/SLIC"]
DEFAULT_MODEL_SPECS = ["GLITCH", "SLAC/SLIC"]


## Duplicated utility removed (single definition of normalize_style_token is kept above)


## Duplicated utility removed (single definition of parse_model_spec is kept above)


def extract_predictions(provider: str, data: dict) -> List[Event]:
    extractor = EXTRACTORS.get(provider)
    if extractor:
        events = extractor(data)
        return events
    # Fallback generic scan when we do not have a provider-specific extractor.
    events: List[Event] = []
    stack: List = [data]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, str):
            events.extend(parse_detection_lines(current))
    return events


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_llm_predictions(responses_dir: Path) -> SystemPredictions:
    systems: SystemPredictions = defaultdict(lambda: defaultdict(set))
    if not responses_dir.exists():
        return systems
    for provider_dir in sorted(p for p in responses_dir.iterdir() if p.is_dir()):
        provider = provider_dir.name
        if provider == "openrouter":
            for model_dir in sorted(p for p in provider_dir.iterdir() if p.is_dir()):
                for tech_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name in TECHS):
                    for file_path in tech_dir.glob("*.json"):
                        data = read_json(file_path)
                        model_name = data.get("model", model_dir.name)
                        style = detect_style_from_filename(file_path.name) or "unknown"
                        system_key = f"{provider}::{model_name}::{style}"
                        events = extract_predictions(provider, data)
                        prompt_target = extract_prompt_target(file_path.stem, style)
                        events = replace_placeholder_paths(events, prompt_target)
                        systems[system_key][tech_dir.name].update(events)
        else:
            for tech_dir in sorted(p for p in provider_dir.iterdir() if p.is_dir() and p.name in TECHS):
                for file_path in tech_dir.glob("*.json"):
                    data = read_json(file_path)
                    model_name = data.get("model")
                    if not model_name:
                        model_name = provider
                    style = detect_style_from_filename(file_path.name) or "unknown"
                    system_key = f"{provider}::{model_name}::{style}"
                    events = extract_predictions(provider, data)
                    prompt_target = extract_prompt_target(file_path.stem, style)
                    events = replace_placeholder_paths(events, prompt_target)
                    systems[system_key][tech_dir.name].update(events)
    return systems


def load_llm_predictions_from_detections(detections_dir: Path) -> SystemPredictions:
    """
    Load LLM predictions from standardized detection CSVs generated by generate_detections.py.
    This provides an alternative to parsing raw response JSONs and is faster for repeated evaluations.
    
    Structure: detections/{method_slug}/{tech}.csv
    Metadata:  detections/{method_slug}/_meta.json
    """
    systems: SystemPredictions = defaultdict(lambda: defaultdict(set))
    
    if not detections_dir.exists():
        return systems
    
    for method_dir in sorted(p for p in detections_dir.iterdir() if p.is_dir()):
        meta_path = method_dir / "_meta.json"
        if not meta_path.exists():
            continue
        
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            continue
        
        # Only load LLM methods (skip oracle, glitch, slac_slic, our)
        if metadata.get("source") != "llm":
            continue
        
        # Extract system key from metadata
        method_key = metadata.get("method_key")
        if not method_key:
            continue
        
        # Load detections for each tech
        for tech in TECHS:
            csv_path = method_dir / f"{tech}.csv"
            if csv_path.exists():
                events = load_csv_events(csv_path, "PATH", "LINE", "CATEGORY")
                systems[method_key][tech] = events
    
    return systems


def compute_metrics(truth: EventSet, preds: Optional[EventSet]) -> Dict[str, Optional[float]]:
    if preds is None:
        return {
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "overlap": None,
        }
    tp = len(truth & preds)
    fp = len(preds - truth)
    fn = len(truth - preds)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not truth and not preds else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not truth else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    denom = tp + fp + fn
    overlap = tp / denom if denom else 1.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap": overlap,
    }


def load_oracle_file_sets(label_dir: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Load all files and positive files (files with at least one smell) from oracle labels.
    Returns: (all_files_per_tech, pos_files_per_tech)
    """
    all_files: Dict[str, Set[str]] = {}
    pos_files: Dict[str, Set[str]] = {}
    
    for tech in TECHS:
        path = label_dir / f"oracle-dataset-{tech}.csv"
        if not path.exists():
            all_files[tech] = set()
            pos_files[tech] = set()
            continue
        
        all_paths: Set[str] = set()
        pos_paths: Set[str] = set()
        
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                file_path = normalize_path(row.get("PATH", ""))
                if not file_path:
                    continue
                all_paths.add(file_path)
                
                category = normalize_category(row.get("CATEGORY", ""))
                if category is not None:
                    pos_paths.add(file_path)
        
        all_files[tech] = all_paths
        pos_files[tech] = pos_paths
    
    return all_files, pos_files


def file_level_accuracy(all_files: Set[str], pos_files: Set[str], preds: Optional[EventSet]) -> Optional[float]:
    """
    Compute file-level accuracy considering TNs (clean files with no predictions).
    
    TP_file: positive file with predictions
    FP_file: negative file with predictions  
    FN_file: positive file without predictions
    TN_file: negative file without predictions
    
    FileAcc = (TP_file + TN_file) / total_files
    """
    if preds is None:
        return None
    
    # Extract unique file paths from predictions
    pred_files = {path for (path, _, _) in preds}
    
    # Negative files are all files minus positive files
    neg_files = all_files - pos_files
    
    # File-level metrics
    tp_file = len(pos_files & pred_files)  # positive files with predictions
    fp_file = len(neg_files & pred_files)  # negative files with predictions
    fn_file = len(pos_files - pred_files)  # positive files without predictions
    tn_file = len(neg_files - pred_files)  # negative files without predictions
    
    total = len(all_files)
    if total == 0:
        return 1.0
    
    return (tp_file + tn_file) / total


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"


def format_int(value: Optional[int]) -> str:
    if value is None:
        return "--"
    return str(value)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dedupe_specs(specs: List[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for spec in specs:
        key = spec.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(spec)
    return ordered


def collect_unique_categories_from_csv(path: Path, column: str) -> Set[str]:
    values: Set[str] = set()
    if not path.exists():
        return values
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get(column, "")
            if raw is None:
                continue
            raw = raw.strip()
            if not raw:
                continue
            values.add(raw)
    return values


def audit_baseline_categories(labels_dir: Path, glitch_dir: Path, slac_slic_dir: Path, intellisa_dir: Path) -> None:
    print("\nCategory audit across baseline sources (raw -> normalized)")
    print("Canonical smells:", ", ".join(SMELL_ORDER))
    for tech in TECHS:
        print(f"\n=== {tech.title()} ===")
        # label
        label_path = labels_dir / f"oracle-dataset-{tech}.csv"
        label_raw = collect_unique_categories_from_csv(label_path, "CATEGORY")
        # glitch
        glitch_path = glitch_dir / f"GLITCH-{tech}-oracle.csv"
        glitch_raw = collect_unique_categories_from_csv(glitch_path, "ERROR")
        # slac/slic
        slac_map = {
            "ansible": slac_slic_dir / "SLAC-ansible-oracle.csv",
            "chef": slac_slic_dir / "SLAC-chef-oracle.csv",
            "puppet": slac_slic_dir / "SLIC-puppet-oracle.csv",
        }
        slac_path = slac_map[tech]
        slac_raw = collect_unique_categories_from_csv(slac_path, "ERROR")
        # intellisa
        intellisa_path = intellisa_dir / f"testset-{tech}.csv"
        intellisa_raw = collect_unique_categories_from_csv(intellisa_path, "CATEGORY")

        def report(source: str, values: Set[str]) -> None:
            unknown: Set[str] = set()
            print(f"{source}: {len(values)} raw categories")
            for v in sorted(values):
                norm = normalize_category(v)
                if norm is None:
                    norm_display = "<none>"
                else:
                    norm_display = norm
                if norm is not None and norm not in SMELL_ORDER:
                    unknown.add(v)
                print(f"  - {v} -> {norm_display}")
            if unknown:
                print(f"  ! Unknown (not in canonical smells after normalization): {sorted(unknown)}")

        report("label", label_raw)
        report("glitch", glitch_raw)
        report("slac_slic", slac_raw)
        report("intellisa", intellisa_raw)


def build_method_lookup(llm_systems: SystemPredictions) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for key in llm_systems.keys():
        display = display_name_for_method(key)
        variants: Set[str] = {display, display.lower(), key, key.lower()}
        parts = key.split("::")
        if len(parts) == 3:
            provider, model, style = parts
            sanitized = sanitize_component(model)
            underscore_slug = model.replace("/", "_")
            style_hyphen = style.replace("_", "-")
            style_variants = {style, style_hyphen}
            model_variants = {
                model,
                underscore_slug,
                sanitized,
            }
            for m in model_variants:
                for s in style_variants:
                    pattern = f"{provider}/{m}-{s}"
                    variants.add(pattern)
                    variants.add(pattern.lower())
        for variant in variants:
            lookup[variant] = key
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate smell detection outputs against oracle labels.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing data/results (default: current directory).",
    )
    parser.add_argument(
        "--models",
        action="append",
        help=(
            "Limit evaluation to the specified set of models. Use 'provider-model-style' (e.g. "
            "'openrouter/x-ai_grok-4-fast:free-definition_based'). Repeat flag to include multiple models. "
            "Special values 'GLITCH' and 'SLAC/SLIC' include the baseline tools."
        ),
    )
    parser.add_argument(
        "--table-format",
        choices=("basic", "occurrence", "full"),
        default="basic",
        help="Select which columns to include in the output tables and CSVs.",
    )
    parser.add_argument(
        "--suffix",
        help=(
            "Optional suffix to append to the metrics CSV files, e.g. '--suffix full' "
            "writes evaluation/metrics_ansible_full.csv."
        ),
    )
    parser.add_argument(
        "--audit-categories",
        action="store_true",
        help="Print a raw->normalized category audit for label/glitch/SLAC_SLIC/intellisa and exit.",
    )
    parser.add_argument(
        "--strict-categories",
        action="store_true",
        help=(
            "Fail if any raw category (post-normalization) is not in canonical 9 smells (ignoring None)."
        ),
    )
    parser.add_argument(
        "--per-smell",
        action="store_true",
        help=(
            "Show per-smell tables (one column per security smell) instead of overall metrics. "
            "Also writes tidy CSVs with full metrics per smell."
        ),
    )
    parser.add_argument(
        "--smell-metric",
        choices=("precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "occurrences"),
        default="f1",
        help=(
            "When using --per-smell, select which metric to display in the table cells. "
            "Full metrics are still written to CSV."
        ),
    )
    parser.add_argument(
        "--use-detections-dir",
        action="store_true",
        help=(
            "Load LLM predictions from standardized detections CSVs (detections/llm/) "
            "instead of parsing raw response JSONs (responses/). This follows the intended pipeline: "
            "run_prompts.py → generate_detections.py → evaluate.py. "
            "Requires running generate_detections.py first."
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    labels_dir = resolve_dir(root, ["data/oracle", "label"])
    glitch_dir = resolve_dir(root, ["data/baselines/glitch", "glitch"])
    slac_slic_dir = resolve_dir(root, ["data/baselines/SLAC_SLIC", "SLAC_SLIC"])
    responses_dir = resolve_dir(root, ["data/responses", "responses"])
    detections_dir = resolve_dir(root, ["data/detections", "detections"])
    intellisa_dir = resolve_dir(root, ["data/intellisa", "intellisa"])
    output_dir = resolve_dir(root, ["results", "evaluation"])

    if args.audit_categories:
        audit_baseline_categories(labels_dir, glitch_dir, slac_slic_dir, intellisa_dir)
        return

    oracle = load_oracle_labels(labels_dir)
    all_files_map, pos_files_map = load_oracle_file_sets(labels_dir)
    glitch = load_glitch_detections(glitch_dir)
    slac_slic = load_slac_slic_detections(slac_slic_dir)
    
    # Load LLM predictions from either detections CSVs or raw response JSONs
    if args.use_detections_dir:
        print(f"Loading LLM predictions from standardized detections: {detections_dir}/")
        llm_systems = load_llm_predictions_from_detections(detections_dir)
        if not llm_systems:
            print(f"Warning: No LLM detections found in {detections_dir}/")
            print("Run 'python generate_detections.py' first to generate standardized detections.")
    else:
        print(f"Loading LLM predictions from raw responses: {responses_dir}/")
        llm_systems = load_llm_predictions(responses_dir)
    
    intellisa_systems = load_intellisa_detections(intellisa_dir)

    ensure_dir(output_dir)

    method_lookup = build_method_lookup(llm_systems)

    if args.models:
        model_specs = dedupe_specs(BASELINE_SPECS + args.models)
    else:
        model_specs = DEFAULT_MODEL_SPECS

    table_format = args.table_format
    format_columns_map = {
        "basic": ["Method", "Precision", "Recall", "F1", "Overlap", "FileAcc"],
        "occurrence": ["Method", "Occurrences", "Precision", "Recall", "F1", "Overlap", "FileAcc"],
        "full": ["Method", "Occurrences", "TP", "FP", "FN", "Precision", "Recall", "F1", "Overlap", "FileAcc"],
    }
    columns = format_columns_map[table_format]
    column_widths = {
        "Method": 40,
        "Occurrences": 12,
        "TP": 6,
        "FP": 6,
        "FN": 6,
        "Precision": 9,
        "Recall": 7,
        "F1": 6,
        "Overlap": 9,
        "FileAcc": 9,
    }

    def resolve_method_spec(spec: str) -> Tuple[str, str, Optional[str]]:
        token = spec.strip()
        upper = token.upper()
        if upper == "GLITCH":
            return ("GLITCH", "glitch", None)
        if upper in {"SLAC", "SLIC", "SLAC/SLIC"}:
            return ("SLAC/SLIC", "slac", None)
        if token in method_lookup:
            key = method_lookup[token]
            return (display_name_for_method(key), "llm", key)
        lower_token = token.lower()
        if lower_token in method_lookup:
            key = method_lookup[lower_token]
            return (display_name_for_method(key), "llm", key)
        parsed_key = parse_model_spec(token)
        if parsed_key and parsed_key in llm_systems:
            return (display_name_for_method(parsed_key), "llm", parsed_key)
        for display, key in method_lookup.items():
            if display.lower() == lower_token:
                return (display_name_for_method(key), "llm", key)
        return (token, "missing", None)

    # Track F1 scores by method across techs for Macro-F1 computation
    f1_by_method: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)

    for tech in TECHS:
        truth = oracle.get(tech, set())
        print(f"\n=== {tech.title()} ===")

        # Pre-split truth by smell to reuse across methods
        truth_by_smell = {smell: {e for e in truth if e[2] == smell} for smell in SMELL_ORDER}

        if args.strict_categories:
            # Validate that all normalized categories in truth and baselines are known
            unknown: Set[str] = set()
            for (path, line, cat) in truth:
                if cat not in SMELL_ORDER:
                    unknown.add(cat)
            for src_name, dataset in (("glitch", glitch.get(tech)), ("slac_slic", slac_slic.get(tech)), ("intellisa", intellisa_systems.get(tech))):
                preds = dataset if dataset is not None else set()
                for (_, _, cat) in preds:
                    if cat not in SMELL_ORDER:
                        unknown.add(cat)
            if unknown:
                raise SystemExit(f"Unknown categories after normalization for {tech}: {sorted(unknown)}")

        if args.per_smell:
            # Per-smell console view: Method + one column per smell with selected metric
            smell_columns = [SMELL_SHORT[s] for s in SMELL_ORDER]
            # Determine widths
            method_width = 40
            cell_width = 11  # fits numbers like 0.000 and "--"

            def print_header() -> None:
                header_parts = [f"{'Method':<{method_width}}"]
                header_parts += [f"{col:>{cell_width}}" for col in smell_columns]
                print(" ".join(header_parts))
                print("-" * (method_width + 1 + len(smell_columns) * (cell_width + 1)))

            def format_metric_cell(smell: str, metrics: Dict[str, Optional[float]], metric_key: str, occ: Optional[int]) -> str:
                if metric_key == "occurrences":
                    return format_int(occ)
                # If no oracle occurrences for this smell, treat precision/recall/F1 as undefined (N/D)
                if len(truth_by_smell[smell]) == 0 and metric_key in {"precision", "recall", "f1"}:
                    return "N/D"
                value = metrics.get(metric_key)
                if metric_key in {"tp", "fp", "fn"}:
                    return format_int(value)  # type: ignore[arg-type]
                return format_float(value)

            def per_smell_metrics(preds: Optional[EventSet]) -> Dict[str, Dict[str, Optional[float]]]:
                if preds is None:
                    # Return Nones for all smells
                    return {smell: compute_metrics(truth_by_smell[smell], None) for smell in SMELL_ORDER}
                by_smell: Dict[str, Dict[str, Optional[float]]] = {}
                for smell in SMELL_ORDER:
                    truth_subset = truth_by_smell[smell]
                    preds_subset = {e for e in preds if e[2] == smell}
                    by_smell[smell] = compute_metrics(truth_subset, preds_subset)
                return by_smell

            def add_and_print_row(method_label: str, preds: Optional[EventSet], csv_rows: List[Dict[str, str]]) -> None:
                by_smell = per_smell_metrics(preds)
                # Compute overall F1 for this tech for Macro-F1 tracking
                overall_metrics = compute_metrics(truth, preds)
                f1_by_method[method_label][tech] = overall_metrics["f1"]
                # Console row
                parts = [f"{method_label:<{method_width}}"]
                for smell in SMELL_ORDER:
                    occ = None if preds is None else len({e for e in preds if e[2] == smell})
                    parts.append(f"{format_metric_cell(smell, by_smell[smell], args.smell_metric, occ):>{cell_width}}")
                print(" ".join(parts))
                # Tidy CSV rows
                for smell in SMELL_ORDER:
                    occ = None if preds is None else len({e for e in preds if e[2] == smell})
                    m = by_smell[smell]
                    # If no oracle occurrences for this smell, output N/D for P/R/F1 in CSV as well
                    no_truth = len(truth_by_smell[smell]) == 0
                    csv_rows.append({
                        "Method": method_label,
                        "Smell": smell,
                        "Occurrences": format_int(occ),
                        "TP": format_int(m["tp"]),
                        "FP": format_int(m["fp"]),
                        "FN": format_int(m["fn"]),
                        "Precision": ("N/D" if no_truth else format_float(m["precision"])),
                        "Recall": ("N/D" if no_truth else format_float(m["recall"])),
                        "F1": ("N/D" if no_truth else format_float(m["f1"])),
                        "Overlap": format_float(m["overlap"]),
                    })

            # Print header
            print_header()
            tidy_rows: List[Dict[str, str]] = []

            # Iterate methods
            for spec in model_specs:
                label, kind, key = resolve_method_spec(spec)
                if kind == "glitch":
                    add_and_print_row(label, glitch.get(tech, set()), tidy_rows)
                elif kind == "slac":
                    add_and_print_row(label, slac_slic.get(tech), tidy_rows)
                elif kind == "llm" and key is not None:
                    preds = llm_systems.get(key, {}).get(tech)
                    add_and_print_row(label, preds, tidy_rows)
                else:
                    add_and_print_row(label, None, tidy_rows)

            # IntelliSA system
            add_and_print_row("IntelliSA", intellisa_systems.get(tech, set()), tidy_rows)

            # Write tidy CSV per tech
            suffix = f"_{args.suffix}" if args.suffix else ""
            out_path = output_dir / f"metrics_{tech}_per_smell{suffix}.csv"
            with out_path.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "Method", "Smell", "Occurrences", "TP", "FP", "FN", "Precision", "Recall", "F1", "Overlap"
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(tidy_rows)
        else:
            # Original overall metrics table
            rows: List[Dict[str, str]] = []

            def format_row_for_print(row: Dict[str, str]) -> str:
                parts = []
                for column in columns:
                    value = row[column]
                    width = column_widths.get(column, 10)
                    align = "<" if column == "Method" else ">"
                    parts.append(f"{value:{align}{width}}")
                return " ".join(parts)

            header_row = {col: col for col in columns}
            print(format_row_for_print(header_row))
            print("-" * sum(column_widths.get(col, 10) + 1 for col in columns))

            def add_row(method_label: str, preds: Optional[EventSet]) -> None:
                metrics = compute_metrics(truth, preds)
                file_acc = file_level_accuracy(all_files_map[tech], pos_files_map[tech], preds)
                occurrence_value = len(preds) if preds is not None else None
                # Store F1 for Macro-F1 calculation
                f1_by_method[method_label][tech] = metrics["f1"]
                row_values = {
                    "Method": method_label,
                    "Occurrences": format_int(occurrence_value) if "Occurrences" in columns else "",
                    "TP": format_int(metrics["tp"]),
                    "FP": format_int(metrics["fp"]),
                    "FN": format_int(metrics["fn"]),
                    "Precision": format_float(metrics["precision"]),
                    "Recall": format_float(metrics["recall"]),
                    "F1": format_float(metrics["f1"]),
                    "Overlap": format_float(metrics["overlap"]),
                    "FileAcc": format_float(file_acc),
                }
                row = {col: row_values[col] if col in row_values else "" for col in columns}
                rows.append(row)
                print(format_row_for_print(row))

            for spec in model_specs:
                label, kind, key = resolve_method_spec(spec)
                if kind == "glitch":
                    add_row(label, glitch.get(tech, set()))
                elif kind == "slac":
                    add_row(label, slac_slic.get(tech))
                elif kind == "llm" and key is not None:
                    preds = llm_systems.get(key, {}).get(tech)
                    add_row(label, preds)
                else:
                    add_row(label, None)

            add_row("IntelliSA", intellisa_systems.get(tech, set()))

            suffix = f"_{args.suffix}" if args.suffix else ""
            output_path = output_dir / f"metrics_{tech}{suffix}.csv"
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)

    # Write Macro-F1 summary CSV
    print("\n=== Macro-F1 Summary ===")
    macro_rows: List[Dict[str, str]] = []
    for method_label in f1_by_method.keys():
        f1_values = f1_by_method[method_label]
        ansible_f1 = f1_values.get("ansible")
        chef_f1 = f1_values.get("chef")
        puppet_f1 = f1_values.get("puppet")
        
        # Compute Macro-F1 as unweighted mean of non-None F1 values
        valid_f1s = [f for f in [ansible_f1, chef_f1, puppet_f1] if f is not None]
        macro_f1 = sum(valid_f1s) / len(valid_f1s) if valid_f1s else None
        
        macro_rows.append({
            "Method": method_label,
            "Ansible": format_float(ansible_f1),
            "Chef": format_float(chef_f1),
            "Puppet": format_float(puppet_f1),
            "Macro-F1": format_float(macro_f1),
        })
        
        # Print to console
        print(f"{method_label:<40} Ansible: {format_float(ansible_f1):>7}  Chef: {format_float(chef_f1):>7}  Puppet: {format_float(puppet_f1):>7}  Macro-F1: {format_float(macro_f1):>7}")
    
    suffix = f"_{args.suffix}" if args.suffix else ""
    macro_path = output_dir / f"metrics_macro{suffix}.csv"
    with macro_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["Method", "Ansible", "Chef", "Puppet", "Macro-F1"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(macro_rows)


if __name__ == "__main__":
    main()
