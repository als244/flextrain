# Backend Requirements

These files are intentionally backend-scoped. Use:

```bash
baseline/scripts/install_backend.sh --backend trl_deepspeed
```

The installer creates `baseline/envs/<backend>`, installs Torch first, installs the selected requirement file, and then installs matching prebuilt FlashAttention 2/3 wheels when available. The harness itself is not installed as `flextrain`; it is used directly from the repo through `PYTHONPATH`.
