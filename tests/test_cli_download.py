"""Tests for first-run download/path helper behaviors.

Focused on the HF auto-download path so example configs can materialize
their local `models/...` directory on first run.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import (  # noqa: E402
    _get_model_flops_per_token,
    _maybe_download_hf_snapshot,
    _normalize_sft_record,
    _resolve_model,
    _resolve_dataset,
    parse_args,
)


def test_maybe_download_hf_snapshot_noop_when_present() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        local_dir = os.path.join(tmp, "models", "already_here")
        os.makedirs(local_dir)
        with open(os.path.join(local_dir, "config.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(local_dir, "model.safetensors"), "wb") as f:
            f.write(b"")
        io_cfg = SimpleNamespace(
            hf_checkpoint=local_dir,
            hf_repo_id="repo/unused",
            hf_revision=None,
        )
        _maybe_download_hf_snapshot(io_cfg)
        assert os.path.isdir(local_dir)


def test_maybe_download_hf_snapshot_downloads_when_missing() -> None:
    calls = []

    def _fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        os.makedirs(kwargs["local_dir"], exist_ok=True)
        with open(os.path.join(kwargs["local_dir"], "config.json"), "w") as f:
            f.write("{}")

    old_mod = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = types.SimpleNamespace(
        snapshot_download=_fake_snapshot_download
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = os.path.join(tmp, "models", "download_me")
            io_cfg = SimpleNamespace(
                hf_checkpoint=local_dir,
                hf_repo_id="repo/example",
                hf_revision="main",
            )
            _maybe_download_hf_snapshot(io_cfg)
            assert os.path.isfile(os.path.join(local_dir, "config.json"))
            assert len(calls) == 1
            assert calls[0]["repo_id"] == "repo/example"
            assert calls[0]["local_dir"] == local_dir
    finally:
        if old_mod is None:
            del sys.modules["huggingface_hub"]
        else:
            sys.modules["huggingface_hub"] = old_mod


def test_resolve_model_local_path() -> None:
    local_dir, repo_id = _resolve_model("models/Llama-3.1-8B")
    assert local_dir == "models/Llama-3.1-8B"
    assert repo_id is None


def test_resolve_model_repo_id() -> None:
    local_dir, repo_id = _resolve_model("org/MyModel")
    assert local_dir == os.path.join("models", "MyModel")
    assert repo_id == "org/MyModel"


def test_get_model_flops_uses_lower_multiplier_for_lora() -> None:
    cfg = SimpleNamespace(
        d_model=8,
        n_heads=2,
        head_dim=4,
        n_kv_heads=1,
        expert_dim=16,
        vocab_size=32,
        n_layers=2,
        top_k=0,
        num_shared_experts=0,
        is_causal=True,
    )
    full = _get_model_flops_per_token(cfg, 10, using_lora=False)
    lora = _get_model_flops_per_token(cfg, 10, using_lora=True)

    active_params_per_layer = (
        2 * cfg.d_model * (cfg.n_heads * cfg.head_dim)
        + 2 * cfg.d_model * (cfg.n_kv_heads * cfg.head_dim)
        + 3 * cfg.d_model * cfg.expert_dim
    )
    expected_delta = (
        cfg.n_layers * (2 * 10 * active_params_per_layer)
        + 2 * 10 * cfg.d_model * cfg.vocab_size
    )
    assert full - lora == expected_delta


def test_parse_args_dataset_flag_kept_simple() -> None:
    args = parse_args(
        [
            "--model", "models/Llama-3.1-8B",
            "--mode", "lora",
            "--max-seq-len", "1024",
            "--max-global-batch-tokens", "2048",
            "--dataset", "my-org/my-sft-dataset",
        ]
    )
    assert args.data_source == "json_sft"
    assert args.dataset == "my-org/my-sft-dataset"


def test_normalize_sft_record_chat_messages() -> None:
    rec = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    normalized = _normalize_sft_record(rec)
    assert normalized is not None
    assert "System:" in normalized["instruction"]
    assert "User:" in normalized["instruction"]
    assert normalized["output"] == "4"


def test_resolve_dataset_existing_local_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.jsonl")
        with open(path, "w") as f:
            f.write("{}\n")
        resolved = _resolve_dataset(path)
        assert resolved == os.path.abspath(path)


def test_resolve_dataset_materializes_hf_dataset() -> None:
    import train as train_mod

    calls = {}
    old_load_dataset = sys.modules.get("datasets")

    def _fake_load_dataset(dataset_name, split):
        calls["dataset_name"] = dataset_name
        calls["split"] = split
        return [
            {"instruction": "Add 1 and 1", "output": "2"},
            {"prompt": "Say hi", "completion": "hi"},
        ]

    sys.modules["datasets"] = types.SimpleNamespace(load_dataset=_fake_load_dataset)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                resolved = train_mod._resolve_dataset("my-org/my-sft-dataset")
                assert os.path.isfile(resolved)
                with open(resolved) as f:
                    lines = [line.strip() for line in f if line.strip()]
                assert len(lines) == 2
                assert calls["dataset_name"] == "my-org/my-sft-dataset"
                assert calls["split"] == "train"
            finally:
                os.chdir(cwd)
    finally:
        if old_load_dataset is None:
            del sys.modules["datasets"]
        else:
            sys.modules["datasets"] = old_load_dataset


def test_main_uses_materialized_json_dataset() -> None:
    import train as train_mod

    calls = {}
    old_maybe_download = train_mod._maybe_download_hf_snapshot
    old_load_cfg = train_mod._load_hf_config_json
    old_select_arch = train_mod.select_arch
    old_arch_module_for = train_mod._arch_module_for
    old_from_pretrained = train_mod.from_pretrained
    old_resolve_dataset = train_mod._resolve_dataset
    old_json_sft_source = train_mod.JsonSFTTokenSource
    old_run_training_loop = train_mod._run_training_loop

    class _FakeArchModule:
        @staticmethod
        def hf_config_to_flextrain(_cfg):
            return {
                "vocab_size": 128256,
                "d_model": 4096,
                "n_heads": 32,
                "head_dim": 128,
                "n_kv_heads": 8,
                "expert_dim": 14336,
                "n_layers": 32,
            }

    class _FakeJsonSFTTokenSource:
        def __init__(self, **kwargs):
            calls["json_source"] = kwargs

    def _fake_from_pretrained(*args, **kwargs):
        calls["from_pretrained"] = kwargs
        return object()

    def _fake_run_training_loop(am, source, **kwargs):
        calls["run_training_loop"] = {
            "am": am,
            "source": source,
            **kwargs,
        }
        return 0

    try:
        train_mod._maybe_download_hf_snapshot = lambda io_cfg: None
        train_mod._load_hf_config_json = lambda model_dir: {
            "architectures": ["LlamaForCausalLM"]
        }
        train_mod.select_arch = lambda hf_cfg: "llama"
        train_mod._arch_module_for = lambda hf_cfg: _FakeArchModule
        train_mod.from_pretrained = _fake_from_pretrained
        train_mod._resolve_dataset = lambda dataset_arg: "/tmp/materialized.jsonl"
        train_mod.JsonSFTTokenSource = _FakeJsonSFTTokenSource
        train_mod._run_training_loop = _fake_run_training_loop

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "models", "Llama-3.1-8B")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "config.json"), "w") as f:
                f.write("{}")

            rc = train_mod.main(
                [
                    "--model", model_dir,
                    "--mode", "lora",
                    "--max-seq-len", "512",
                    "--max-global-batch-tokens", "1024",
                    "--dataset", "my-org/my-sft-dataset",
                ]
            )

        assert rc == 0
        assert calls["json_source"]["path"] == "/tmp/materialized.jsonl"
        assert calls["json_source"]["tokenizer"] == model_dir
        assert calls["json_source"]["max_seq_len"] == 512
        assert calls["run_training_loop"]["source"].__class__ is _FakeJsonSFTTokenSource
    finally:
        train_mod._maybe_download_hf_snapshot = old_maybe_download
        train_mod._load_hf_config_json = old_load_cfg
        train_mod.select_arch = old_select_arch
        train_mod._arch_module_for = old_arch_module_for
        train_mod.from_pretrained = old_from_pretrained
        train_mod._resolve_dataset = old_resolve_dataset
        train_mod.JsonSFTTokenSource = old_json_sft_source
        train_mod._run_training_loop = old_run_training_loop


def _run_all() -> None:
    tests = [
        (
            "test_maybe_download_hf_snapshot_noop_when_present",
            test_maybe_download_hf_snapshot_noop_when_present,
        ),
        (
            "test_maybe_download_hf_snapshot_downloads_when_missing",
            test_maybe_download_hf_snapshot_downloads_when_missing,
        ),
        (
            "test_resolve_model_local_path",
            test_resolve_model_local_path,
        ),
        (
            "test_resolve_model_repo_id",
            test_resolve_model_repo_id,
        ),
        (
            "test_get_model_flops_uses_lower_multiplier_for_lora",
            test_get_model_flops_uses_lower_multiplier_for_lora,
        ),
        (
            "test_parse_args_dataset_flag_kept_simple",
            test_parse_args_dataset_flag_kept_simple,
        ),
        (
            "test_normalize_sft_record_chat_messages",
            test_normalize_sft_record_chat_messages,
        ),
        (
            "test_resolve_dataset_existing_local_path",
            test_resolve_dataset_existing_local_path,
        ),
        (
            "test_resolve_dataset_materializes_hf_dataset",
            test_resolve_dataset_materializes_hf_dataset,
        ),
        (
            "test_main_uses_materialized_json_dataset",
            test_main_uses_materialized_json_dataset,
        ),
    ]
    for name, fn in tests:
        print(f"  - {name}")
        fn()
    print(f"  OK ({len(tests)} tests)")


if __name__ == "__main__":
    _run_all()
