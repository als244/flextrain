# Backend Requirements

These files are intentionally backend-scoped. One per backend:
`trl_deepspeed.txt`, `trl_fsdp.txt`, `deepspeed_arctic.txt`,
`megatrain.txt`, `torchtitan.txt`, `megatron.txt`.

Install with:

```bash
baseline/scripts/install_backend.sh --backend trl_fsdp
```

The installer creates `baseline/envs/<backend>`, installs Torch first
(via `--torch-index-url`, default `cu126`, `auto` to detect from local
CUDA), installs the selected requirement file, then installs matching
prebuilt FlashAttention 2/3 wheels when available. The harness itself
is not installed as `flextrain`; it is used directly from the repo
through `PYTHONPATH`.
