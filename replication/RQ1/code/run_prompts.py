import argparse
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# Prefer the first existing candidate under the given root; otherwise return the first candidate.
def resolve_dir(root: Path, path: Path, fallbacks: Sequence[str]) -> Path:
    if path.is_absolute():
        return path
    primary = root / path
    if primary.exists():
        return primary
    for rel in fallbacks:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return primary

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - optional dependency
    Anthropic = None

try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import system as xai_system_message
    from xai_sdk.chat import user as xai_user_message
except ImportError:  # pragma: no cover - optional dependency
    XAIClient = None
    xai_user_message = None
    xai_system_message = None

try:
    import grpc  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    grpc = None


HEADER_DELIMITER = "# ============================================================"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_XAI = "xai"
PROVIDER_OPENROUTER = "openrouter"
DEFAULT_MODELS = {
    PROVIDER_OPENAI: "gpt-5-2025-08-07",
    PROVIDER_ANTHROPIC: "claude-sonnet-4-20250514",
    PROVIDER_XAI: "grok-4-fast-non-reasoning",
    PROVIDER_OPENROUTER: "openrouter/anthropic/claude-3.5-sonnet",
}
PROVIDER_ENV_VARS = {
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
    PROVIDER_XAI: "XAI_API_KEY",
    PROVIDER_OPENROUTER: "OPENAI_COMPATIBLE_API_KEY",
}
SUPPORTED_STYLES = (
    "definition_based",
    "static_analysis_rules",
)

OPENAI_BATCH_ENDPOINT = "/v1/responses"
OPENAI_BATCH_SUCCESS_STATUS = "completed"
OPENAI_BATCH_TERMINAL_STATUSES = {"failed", "cancelled", "completed", "expired"}
OPENAI_BATCH_COMPLETION_WINDOW = "24h"


def normalize_style_arg(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    token = value.strip().lower().replace("-", "_")
    mapping = {
        "definition": "definition_based",
        "definition_based": "definition_based",
        "static": "static_analysis_rules",
        "static_analysis_rules": "static_analysis_rules",
    }
    return mapping.get(token, value)


def sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    cleaned = cleaned.strip("_")
    return cleaned or "model"


def iter_prompt_files(
    prompts_dir: Path,
    style: Optional[str],
    includes: Optional[Sequence[str]],
) -> Iterable[Path]:
    files = sorted(
        path
        for path in prompts_dir.rglob("*.txt")
        if path.is_file()
    )
    for path in files:
        relative_str = str(path.relative_to(prompts_dir))
        if style and not path.name.startswith(f"prompt_{style}_"):
            continue
        if includes and not any(token in relative_str for token in includes):
            continue
        yield path


def load_prompt(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if HEADER_DELIMITER in raw:
        _, _, raw = raw.partition(HEADER_DELIMITER)
    return raw.lstrip("\r\n").strip()


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def write_response(path: Path, response: Any) -> None:
    ensure_output_dir(path.parent)
    path.write_text(json.dumps(response, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def as_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [as_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): as_serializable(val) for key, val in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "dict"):
        return value.dict()
    return json.loads(json.dumps(value, default=str))


def build_client(provider: str, api_key: str):
    if provider == PROVIDER_OPENAI:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Run 'pip install openai'.")
        return OpenAI(api_key=api_key)
    if provider == PROVIDER_ANTHROPIC:
        if Anthropic is None:
            raise RuntimeError("anthropic package is not installed. Run 'pip install anthropic'.")
        return Anthropic(api_key=api_key)
    if provider == PROVIDER_XAI:
        if XAIClient is None:
            raise RuntimeError("xai-sdk package is not installed. Run 'pip install xai-sdk'.")
        return XAIClient(api_key=api_key, timeout=3600)
    if provider == PROVIDER_OPENROUTER:
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        headers = {}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.environ.get("OPENROUTER_X_TITLE")
        if title:
            headers["X-Title"] = title
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers or None,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def resolve_api_key(provider: str) -> str:
    env_var = PROVIDER_ENV_VARS[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(f"{env_var} environment variable is not set for provider '{provider}'.")
    return api_key


def call_model(
    provider: str,
    client,
    model: str,
    prompt: str,
    max_output_tokens: int,
    openrouter_reasoning_effort: Optional[str],
) -> Any:
    if provider == PROVIDER_OPENAI:
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
        return as_serializable(response)
    if provider == PROVIDER_OPENROUTER:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        extra_headers = {}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        if referer:
            extra_headers["HTTP-Referer"] = referer
        title = os.environ.get("OPENROUTER_X_TITLE")
        if title:
            extra_headers["X-Title"] = title

        extra_body = None
        if openrouter_reasoning_effort:
            extra_body = {"reasoning": {"effort": openrouter_reasoning_effort}}

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers=extra_headers or None,
            extra_body=extra_body,
        )
        return as_serializable(response)
    if provider == PROVIDER_ANTHROPIC:
        response = client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return as_serializable(response)
    if provider == PROVIDER_XAI:
        if xai_user_message is None or xai_system_message is None:
            raise RuntimeError("xai-sdk package is not installed. Run 'pip install xai-sdk'.")
        chat = client.chat.create(model=model)
        chat.append(xai_system_message("You are Grok, a highly intelligent, helpful AI assistant."))
        chat.append(xai_user_message(prompt))
        try:
            response = chat.sample()
        except Exception as err:  # pragma: no cover - network interaction
            if grpc and isinstance(err, grpc.RpcError):
                details = err.details() if hasattr(err, "details") else str(err)
                raise RuntimeError(
                    f"Grok request failed: {details}. Verify that XAI_API_KEY is correct and has access to {model}."
                ) from err
            raise
        return as_serializable(response)
    raise ValueError(f"Unsupported provider: {provider}")


def build_openai_batch_custom_id(relative_path: Path) -> str:
    canonical = relative_path.as_posix()
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", canonical).strip("_") or "item"
    max_length = 80
    if len(sanitized) > max_length:
        sanitized = sanitized[-max_length:]
    return f"{sanitized}__{digest}"


def write_json_file(path: Path, payload: Any) -> None:
    ensure_output_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl_file(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    ensure_output_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def parse_jsonl_records(payload: str) -> Iterator[Dict[str, Any]]:
    for line in payload.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            preview = line[:120]
            print(f"Warning: unable to decode JSONL line: {preview}")


def read_file_content(file_response: Any) -> str:
    if file_response is None:
        return ""
    if hasattr(file_response, "text"):
        text = file_response.text
        if callable(text):  # type: ignore[call-arg]
            text = text()
        if isinstance(text, bytes):
            return text.decode("utf-8")
        return str(text)
    if hasattr(file_response, "read"):
        data = file_response.read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)
    return str(file_response)


def parse_metadata_args(values: Optional[Sequence[str]]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    if not values:
        return metadata
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Batch metadata '{raw}' is not in key=value format.")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("Batch metadata keys cannot be empty.")
        metadata[key] = value
    return metadata


def prepare_openai_batch_requests(
    jobs: Sequence[Dict[str, Any]],
    model: str,
    max_output_tokens: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    requests: List[Dict[str, Any]] = []
    mapping_entries: List[Dict[str, Any]] = []
    seen_custom_ids: Set[str] = set()

    for job in jobs:
        relative_path: Path = job["relative_path"]
        custom_id = build_openai_batch_custom_id(relative_path)
        base_custom_id = custom_id
        suffix = 1
        while custom_id in seen_custom_ids:
            custom_id = f"{base_custom_id}_{suffix}"
            suffix += 1
        seen_custom_ids.add(custom_id)

        body: Dict[str, Any] = {
            "model": model,
            "input": job["prompt"],
        }
        if max_output_tokens:
            body["max_output_tokens"] = max_output_tokens

        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": OPENAI_BATCH_ENDPOINT,
                "body": body,
            }
        )
        mapping_entries.append(
            {
                "custom_id": custom_id,
                "prompt_path": str(job["prompt_path"]),
                "relative_prompt_path": relative_path.as_posix(),
                "output_path": str(job["output_path"]),
            }
        )

    return requests, mapping_entries


def await_openai_batch_completion(
    client,
    batch_id: str,
    poll_interval: float,
    timeout: Optional[float],
) -> Tuple[Any, bool]:
    poll_interval = max(1.0, poll_interval)
    start_time = time.time()
    batch = client.batches.retrieve(batch_id)
    while True:
        status = getattr(batch, "status", "unknown")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{timestamp}] Batch {batch_id} status: {status}")
        if status in OPENAI_BATCH_TERMINAL_STATUSES:
            return batch, False
        if timeout is not None and (time.time() - start_time) >= timeout:
            return batch, True
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch_id)


def load_openai_batch_mapping(mapping_path: Path) -> Dict[str, Path]:
    if not mapping_path.exists():
        return {}
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping: Dict[str, Path] = {}
    for item in data.get("requests", []):
        custom_id = item.get("custom_id")
        output_path = item.get("output_path")
        if not custom_id or not output_path:
            continue
        mapping[custom_id] = Path(output_path)
    return mapping


def download_openai_batch_outputs(
    client,
    batch: Any,
    artifacts_dir: Path,
    id_to_output: Dict[str, Path],
) -> None:
    batch_id = getattr(batch, "id", None)
    if not batch_id:
        print("Batch identifier missing; cannot download results.")
        return

    output_file_id = getattr(batch, "output_file_id", None)
    if output_file_id:
        response = client.files.content(output_file_id)
        content = read_file_content(response)
        output_jsonl_path = artifacts_dir / f"{batch_id}_output.jsonl"
        output_jsonl_path.write_text(content, encoding="utf-8")
        for record in parse_jsonl_records(content):
            custom_id = record.get("custom_id")
            if not custom_id:
                continue
            response_section = record.get("response")
            if not isinstance(response_section, dict):
                print(f"Warning: response payload missing for custom_id {custom_id}")
                continue
            body = response_section.get("body")
            if body is None:
                print(f"Warning: response body missing for custom_id {custom_id}")
                continue
            output_path = id_to_output.get(custom_id)
            if not output_path:
                print(f"Warning: no mapping found for custom_id {custom_id}; skipping write.")
                continue
            write_response(output_path, as_serializable(body))
            print(f"Saved response for {custom_id} -> {output_path}")
    else:
        print(f"Batch {batch_id} has no output file yet; skipping response download.")

    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        response = client.files.content(error_file_id)
        content = read_file_content(response)
        error_jsonl_path = artifacts_dir / f"{batch_id}_errors.jsonl"
        error_jsonl_path.write_text(content, encoding="utf-8")
        for record in parse_jsonl_records(content):
            custom_id = record.get("custom_id", "<unknown>")
            error_info = record.get("error")
            if error_info:
                print(f"Batch error for {custom_id}: {error_info}")
    else:
        print(f"Batch {batch_id} reported no error file.")


def run_openai_batch(
    prompts_dir: Path,
    target_dir: Path,
    client,
    model: str,
    overwrite: bool,
    max_output_tokens: int,
    style: Optional[str],
    small_batch: bool,
    includes: Optional[Sequence[str]],
    batch_id: Optional[str],
    batch_wait: bool,
    batch_poll_interval: float,
    batch_timeout: Optional[float],
    batch_metadata_args: Optional[Sequence[str]],
) -> None:
    ensure_output_dir(target_dir)
    artifacts_dir = target_dir / "batches"
    ensure_output_dir(artifacts_dir)

    if batch_id:
        batch_id = batch_id.strip()
    if batch_id:
        print(f"Retrieving existing batch {batch_id}...")
        try:
            batch = client.batches.retrieve(batch_id)
        except Exception as exc:  # pragma: no cover - network
            raise RuntimeError(f"Failed to retrieve batch {batch_id}: {exc}") from exc

        status = getattr(batch, "status", "unknown")
        print(f"Batch {batch_id} current status: {status}")
        if batch_wait and status not in OPENAI_BATCH_TERMINAL_STATUSES:
            print(f"Waiting for batch {batch_id} to reach a terminal status...")
            batch, timed_out = await_openai_batch_completion(
                client,
                batch_id=batch_id,
                poll_interval=batch_poll_interval,
                timeout=batch_timeout,
            )
            if timed_out:
                elapsed = batch_timeout if batch_timeout is not None else "unknown"
                print(f"Timeout reached ({elapsed} seconds); latest status: {getattr(batch, 'status', 'unknown')}")
                write_json_file(artifacts_dir / f"{batch_id}_batch.json", as_serializable(batch))
                return
            print(f"Batch {batch_id} finished with status {getattr(batch, 'status', 'unknown')}.")
        write_json_file(artifacts_dir / f"{batch_id}_batch.json", as_serializable(batch))

        mapping = load_openai_batch_mapping(artifacts_dir / f"{batch_id}_mapping.json")
        if not mapping:
            print(f"Warning: no mapping file found for batch {batch_id}; responses may not be written.")
        if getattr(batch, "status", None) == OPENAI_BATCH_SUCCESS_STATUS:
            download_openai_batch_outputs(client, batch, artifacts_dir, mapping)
        else:
            print(f"Batch {batch_id} is in status '{getattr(batch, 'status', 'unknown')}'. Skipping download.")
        return

    try:
        metadata = parse_metadata_args(batch_metadata_args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    jobs: List[Dict[str, Any]] = []
    processed_counts: Dict[str, int] = {}

    for prompt_path in iter_prompt_files(prompts_dir, style=style, includes=includes):
        relative_path = prompt_path.relative_to(prompts_dir)
        tech = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"

        if small_batch and not includes:
            if processed_counts.get(tech, 0) >= 3:
                continue

        output_path = target_dir / relative_path.with_suffix(".json")

        if output_path.exists() and not overwrite:
            print(f"Skipping {prompt_path.name}: response already exists at {output_path}")
            continue

        prompt = load_prompt(prompt_path)
        if not prompt:
            print(f"Skipping {prompt_path.name}: prompt is empty")
            continue

        jobs.append(
            {
                "prompt_path": prompt_path,
                "relative_path": relative_path,
                "output_path": output_path,
                "prompt": prompt,
            }
        )
        print(f"Queued {prompt_path.name} for batch processing (output -> {output_path})")

        if small_batch and not includes:
            processed_counts[tech] = processed_counts.get(tech, 0) + 1

    if not jobs:
        print("No prompts to enqueue for batch processing.")
        return

    requests, mapping_entries = prepare_openai_batch_requests(
        jobs=jobs,
        model=model,
        max_output_tokens=max_output_tokens,
    )

    temp_input_path = artifacts_dir / f"{uuid.uuid4().hex}_input.jsonl"
    write_jsonl_file(temp_input_path, requests)

    print(f"Uploading batch input file ({len(requests)} requests)...")
    try:
        with temp_input_path.open("rb") as handle:
            batch_input_file = client.files.create(file=handle, purpose="batch")
    except Exception as exc:  # pragma: no cover - network interaction
        raise RuntimeError(f"Failed to upload batch input file: {exc}") from exc

    batch_input_file_id = getattr(batch_input_file, "id", None)
    if not batch_input_file_id:
        raise RuntimeError("OpenAI did not return an id for the uploaded batch input file.")

    print("Creating batch job...")
    try:
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint=OPENAI_BATCH_ENDPOINT,
            completion_window=OPENAI_BATCH_COMPLETION_WINDOW,
            metadata=metadata or None,
        )
    except Exception as exc:  # pragma: no cover - network interaction
        raise RuntimeError(f"Failed to create batch job: {exc}") from exc

    batch_id = getattr(batch, "id", None)
    if not batch_id:
        raise RuntimeError("OpenAI did not return a batch id.")

    final_input_path = artifacts_dir / f"{batch_id}_input.jsonl"
    try:
        temp_input_path.rename(final_input_path)
    except FileExistsError:
        final_input_path.unlink()
        temp_input_path.rename(final_input_path)

    write_json_file(artifacts_dir / f"{batch_id}_mapping.json", {
        "batch_id": batch_id,
        "provider": PROVIDER_OPENAI,
        "model": model,
        "endpoint": OPENAI_BATCH_ENDPOINT,
        "max_output_tokens": max_output_tokens,
        "request_count": len(requests),
        "requests": mapping_entries,
    })
    write_json_file(artifacts_dir / f"{batch_id}_batch.json", as_serializable(batch))
    write_json_file(artifacts_dir / f"{batch_id}_input_file.json", as_serializable(batch_input_file))

    print(f"Batch {batch_id} created with {len(requests)} requests.")
    print(f"Input file stored at {final_input_path}")
    print(f"Mapping file stored at {artifacts_dir / f'{batch_id}_mapping.json'}")

    if batch_wait:
        print(f"Waiting for batch {batch_id} to reach a terminal status...")
        batch, timed_out = await_openai_batch_completion(
            client,
            batch_id=batch_id,
            poll_interval=batch_poll_interval,
            timeout=batch_timeout,
        )
        write_json_file(artifacts_dir / f"{batch_id}_batch.json", as_serializable(batch))
        if timed_out:
            elapsed = batch_timeout if batch_timeout is not None else "unknown"
            print(f"Timeout reached ({elapsed} seconds); latest status: {getattr(batch, 'status', 'unknown')}")
            return
        print(f"Batch {batch_id} finished with status {getattr(batch, 'status', 'unknown')}.")
        mapping = {entry["custom_id"]: Path(entry["output_path"]) for entry in mapping_entries}
        if getattr(batch, "status", None) == OPENAI_BATCH_SUCCESS_STATUS:
            download_openai_batch_outputs(client, batch, artifacts_dir, mapping)
        else:
            print(f"Batch {batch_id} did not complete successfully (status: {getattr(batch, 'status', 'unknown')}).")
    else:
        print("Use --batch-wait or rerun with --batch-id to monitor and download results once ready.")


def run(
    prompts_dir: Path,
    output_dir: Path,
    provider: str,
    model: str,
    overwrite: bool,
    max_output_tokens: int,
    style: Optional[str],
    small_batch: bool,
    includes: Optional[Sequence[str]],
    openrouter_reasoning_effort: Optional[str],
    use_openai_batch: bool,
    openai_batch_id: Optional[str],
    batch_wait: bool,
    batch_poll_interval: float,
    batch_timeout: Optional[float],
    batch_metadata_args: Optional[Sequence[str]],
) -> None:
    prompts_dir = prompts_dir.resolve()
    output_dir = output_dir.resolve()
    ensure_output_dir(output_dir)

    provider = provider.lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown provider: {provider}")

    target_dir = output_dir / provider
    if provider == PROVIDER_OPENROUTER:
        target_dir = target_dir / sanitize_component(model)
    ensure_output_dir(target_dir)

    api_key = resolve_api_key(provider)
    client = build_client(provider, api_key)

    if use_openai_batch:
        if provider != PROVIDER_OPENAI:
            raise ValueError("--use-openai-batch is only supported when provider is 'openai'.")
        run_openai_batch(
            prompts_dir=prompts_dir,
            target_dir=target_dir,
            client=client,
            model=model,
            overwrite=overwrite,
            max_output_tokens=max_output_tokens,
            style=style,
            small_batch=small_batch,
            includes=includes,
            batch_id=openai_batch_id,
            batch_wait=batch_wait,
            batch_poll_interval=batch_poll_interval,
            batch_timeout=batch_timeout,
            batch_metadata_args=batch_metadata_args,
        )
        return

    processed_counts: Dict[str, int] = {}

    for prompt_path in iter_prompt_files(prompts_dir, style=style, includes=includes):
        relative_path = prompt_path.relative_to(prompts_dir)
        tech = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"

        if small_batch and not includes:
            if processed_counts.get(tech, 0) >= 3:
                continue

        output_path = target_dir / relative_path.with_suffix(".json")

        if output_path.exists() and not overwrite:
            print(f"Skipping {prompt_path.name}: response already exists at {output_path}")
            continue

        prompt = load_prompt(prompt_path)
        if not prompt:
            print(f"Skipping {prompt_path.name}: prompt is empty")
            continue

        print(f"Processing {prompt_path.name} with {provider}:{model}...")
        response = call_model(
            provider=provider,
            client=client,
            model=model,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            openrouter_reasoning_effort=openrouter_reasoning_effort,
        )
        write_response(output_path, response)
        print(f"Saved response to {output_path}")

        if small_batch and not includes:
            processed_counts[tech] = processed_counts.get(tech, 0) + 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch prompts through configurable LLM providers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = parser.add_argument_group("I/O")
    io.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing data/output (default: current directory).",
    )
    io.add_argument(
        "--prompts",
        type=Path,
        default=Path("prompts"),
        help="Directory containing .txt prompt files",
    )
    io.add_argument(
        "--output",
        type=Path,
        default=Path("responses"),
        help="Directory where JSON responses will be written",
    )
    io.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON responses",
    )

    provider = parser.add_argument_group("Provider")
    provider.add_argument(
        "--provider",
        choices=sorted(DEFAULT_MODELS.keys()),
        default=PROVIDER_OPENAI,
        help="LLM provider to use for generation",
    )
    provider.add_argument(
        "--model",
        help="Model name to send with the request. Defaults vary by provider.",
    )
    provider.add_argument(
        "--openrouter-reasoning-effort",
        choices=["low", "medium", "high"],
        help="When using OpenRouter, set the reasoning effort to adjust (cannot fully disable reasoning).",
    )

    openai_batch = parser.add_argument_group("OpenAI Batch")
    openai_batch.add_argument(
        "--use-openai-batch",
        action="store_true",
        help="Use the OpenAI Batch API for asynchronous processing (provider must be 'openai').",
    )
    openai_batch.add_argument(
        "--batch-id",
        help="Existing OpenAI batch id to monitor or download results for (requires --use-openai-batch).",
    )
    openai_batch.add_argument(
        "--batch-wait",
        action="store_true",
        help="When creating or retrieving a batch, poll until it reaches a terminal status.",
    )
    openai_batch.add_argument(
        "--batch-poll-interval",
        type=float,
        default=30.0,
        help="Seconds between batch status polls when --batch-wait is used.",
    )
    openai_batch.add_argument(
        "--batch-timeout",
        type=float,
        help="Maximum number of seconds to wait when --batch-wait is enabled.",
    )
    openai_batch.add_argument(
        "--batch-metadata",
        action="append",
        help="Attach custom metadata to new batches as key=value pairs (can be repeated).",
    )

    filtering = parser.add_argument_group("Filtering")
    filtering.add_argument(
        "--style",
        choices=SUPPORTED_STYLES,
        help="Filter prompts by style when processing nested directories",
    )
    filtering.add_argument(
        "--include", "--includes",
        action="append",
        dest="include",
        help="Only process prompts whose relative path contains this substring (can be repeated).",
    )
    filtering.add_argument(
        "--small-batch",
        action="store_true",
        help="Process only the first three prompts per tech directory (definition vs static).",
    )

    gen = parser.add_argument_group("Generation")
    gen.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Maximum number of output tokens to request from the model",
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    # Normalize style synonyms to supported choices
    args.style = normalize_style_arg(args.style)
    root = args.root.resolve()
    prompts_dir = resolve_dir(root, args.prompts, ["data/prompts"])
    output_dir = resolve_dir(root, args.output, ["data/responses", "responses"])
    model = args.model or DEFAULT_MODELS[args.provider]
    run(
        prompts_dir=prompts_dir,
        output_dir=output_dir,
        provider=args.provider,
        model=model,
        overwrite=args.overwrite,
        max_output_tokens=args.max_output_tokens,
        style=args.style,
        small_batch=args.small_batch,
        includes=args.include,
        openrouter_reasoning_effort=args.openrouter_reasoning_effort,
        use_openai_batch=args.use_openai_batch,
        openai_batch_id=args.batch_id,
        batch_wait=args.batch_wait,
        batch_poll_interval=args.batch_poll_interval,
        batch_timeout=args.batch_timeout,
        batch_metadata_args=args.batch_metadata,
    )
