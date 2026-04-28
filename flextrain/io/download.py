"""HF model / dataset download helpers.

Two functions, both safe to call from compute nodes that DO have internet
(or any environment) so the resulting paths can be passed to
:func:`flextrain.api.from_pretrained` from compute nodes that DON'T:

* :func:`download_model` -- mirror an HF model repo to a local directory.
  Wraps ``huggingface_hub.snapshot_download`` with explicit target +
  completeness check.
* :func:`download_dataset` -- materialize an HF dataset to a local
  ``.jsonl`` file in FlexTrain's SFT schema (instruction/output/input).
  Same normalization rules as ``train.py``'s in-flight materialization,
  factored out so the ``download.py`` CLI and ``train.py`` share one
  source of truth.

The intended workflow on an air-gapped cluster:

    # On a login node with internet access:
    python download.py model meta-llama/Llama-3.1-8B --target models/Llama-3.1-8B
    python download.py dataset HuggingFaceH4/no_robots --target datasets/no_robots.jsonl

    # Then on the compute node (no internet):
    python train.py --model models/Llama-3.1-8B \
                    --data-source json_sft --dataset datasets/no_robots.jsonl ...
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Model snapshots (HF safetensors)
# ---------------------------------------------------------------------------


def hf_checkpoint_is_complete(local_path: str | os.PathLike) -> bool:
    """True if ``local_path`` already holds a fully-downloaded HF checkpoint
    (``config.json`` + either a single ``model.safetensors`` OR a sharded
    index whose every referenced shard exists on disk).

    We prefer this over ``os.path.exists`` because ``snapshot_download``
    leaves partial downloads on the filesystem when interrupted, and a
    later run shouldn't silently use them.
    """
    local_path = str(local_path)
    cfg_path = os.path.join(local_path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    single_path = os.path.join(local_path, "model.safetensors")
    if os.path.isfile(single_path):
        return True
    index_path = os.path.join(local_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return False
    with open(index_path) as f:
        index_payload = json.load(f)
    weight_map = index_payload.get("weight_map", {})
    if not weight_map:
        return False
    shard_files = {str(name) for name in weight_map.values()}
    return all(
        os.path.isfile(os.path.join(local_path, shard_name))
        for shard_name in shard_files
    )


def download_model(
    repo_id: str,
    target_dir: str | os.PathLike,
    *,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> str:
    """Snapshot ``repo_id`` to ``target_dir``. Returns the absolute target path.

    No-op if ``target_dir`` already passes :func:`hf_checkpoint_is_complete`
    and ``force`` is False -- so this is safe to re-run on a login node
    that lost its connection partway through.
    """
    target = Path(target_dir).resolve()
    if not force and hf_checkpoint_is_complete(target):
        if verbose:
            print(f"[download] {repo_id} already complete at {target}", flush=True)
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "download_model needs `huggingface_hub`. "
            "Install via `pip install huggingface_hub`."
        ) from e

    if verbose:
        print(f"[download] fetching {repo_id} -> {target}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    if verbose:
        print(f"[download] done: {repo_id} at {target}", flush=True)
    return str(target)


# ---------------------------------------------------------------------------
# Dataset materialization (HF datasets -> FlexTrain SFT JSONL)
# ---------------------------------------------------------------------------


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Iterable) and not isinstance(
        content, (bytes, bytearray, dict)
    ):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text = str(item.get("text", "")).strip()
                else:
                    text = ""
            else:
                text = ""
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _normalize_chat_record(rec: dict[str, Any]) -> dict[str, str] | None:
    messages = rec.get("messages") or rec.get("conversations")
    if not isinstance(messages, list):
        return None

    normalized: list[tuple[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "") or msg.get("from", "")).strip().lower()
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        text = _flatten_message_content(msg.get("content", msg.get("value")))
        if role and text:
            normalized.append((role, text))

    last_assistant = None
    for i in range(len(normalized) - 1, -1, -1):
        if normalized[i][0] == "assistant":
            last_assistant = i
            break
    if last_assistant is None or last_assistant == 0:
        return None

    role_names = {"system": "System", "user": "User", "assistant": "Assistant"}
    prompt_lines = [
        f"{role_names.get(role, role.title())}:\n{text}"
        for role, text in normalized[:last_assistant]
    ]
    response = normalized[last_assistant][1].strip()
    if not prompt_lines or not response:
        return None
    return {
        "instruction": "\n\n".join(prompt_lines),
        "output": response,
        "input": "",
    }


def normalize_sft_record(rec: Any) -> dict[str, str] | None:
    """Coerce one HF dataset record into FlexTrain's
    ``{"instruction", "output", "input"}`` SFT schema. Tries common
    column-pair conventions first (instruction/output, prompt/completion,
    question/answer, ...) then falls back to chat-style ``messages``."""
    if not isinstance(rec, dict):
        return None
    candidates = [
        ("instruction", "output", "input"),
        ("prompt", "completion", "input"),
        ("prompt", "response", "input"),
        ("question", "answer", "context"),
        ("query", "response", "context"),
    ]
    for prompt_key, response_key, input_key in candidates:
        prompt = str(rec.get(prompt_key, "") or "").strip()
        response = str(rec.get(response_key, "") or "").strip()
        if prompt and response:
            return {
                "instruction": prompt,
                "output": response,
                "input": str(rec.get(input_key, "") or "").strip(),
            }
    return _normalize_chat_record(rec)


def download_dataset(
    dataset_spec: str,
    target_path: str | os.PathLike,
    *,
    split: str = "train",
    config: str | None = None,
    force: bool = False,
    verbose: bool = True,
) -> str:
    """Materialize an HF dataset (``org/name``) or remote URL to a local
    JSONL file in FlexTrain's SFT schema. Returns the absolute target path.

    Modes:

    * ``dataset_spec`` is an existing local file -> return its absolute path.
    * ``dataset_spec`` starts with ``http://`` / ``https://`` -> download
      the file as-is to ``target_path`` (no schema normalization; assumes
      the user is fetching a JSONL already in the right format).
    * Otherwise -> ``datasets.load_dataset(dataset_spec, split=...)`` and
      run :func:`normalize_sft_record` over each row, writing JSONL to
      ``target_path``.

    No-op if ``target_path`` already exists and ``force`` is False.
    """
    target = Path(target_path).resolve()

    # Local file already on disk -- nothing to do.
    if os.path.isfile(dataset_spec):
        return os.path.abspath(dataset_spec)

    if not force and target.is_file():
        if verbose:
            print(
                f"[download] dataset already at {target}", flush=True,
            )
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)

    # Remote URL -> raw fetch.
    if dataset_spec.startswith(("http://", "https://")):
        try:
            from urllib.request import urlretrieve
        except ImportError as e:  # pragma: no cover
            raise ImportError("Could not import urllib.request") from e
        if verbose:
            print(f"[download] fetching {dataset_spec} -> {target}", flush=True)
        urlretrieve(dataset_spec, str(target))
        return str(target)

    # HF dataset -> load + normalize.
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "download_dataset needs `datasets`. "
            "Install via `pip install datasets`."
        ) from e

    if verbose:
        cfg_str = f", config={config}" if config else ""
        print(
            f"[download] loading HF dataset {dataset_spec} (split={split}{cfg_str})...",
            flush=True,
        )
    if config is not None:
        ds = load_dataset(dataset_spec, config, split=split)
    else:
        ds = load_dataset(dataset_spec, split=split)

    kept = 0
    skipped = 0
    with target.open("w") as f:
        for rec in ds:
            normalized = normalize_sft_record(rec)
            if normalized is None:
                skipped += 1
                continue
            f.write(json.dumps(normalized) + "\n")
            kept += 1

    if kept == 0:
        target.unlink(missing_ok=True)
        raise ValueError(
            f"Could not build SFT examples from {dataset_spec!r}. "
            "Expected records like instruction/output, prompt/completion, "
            "question/answer, or chat-style messages."
        )
    if verbose:
        print(
            f"[download] wrote {kept} records ({skipped} skipped) "
            f"-> {target}",
            flush=True,
        )
    return str(target)
