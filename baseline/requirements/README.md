# Backend Requirements

Two consolidated, env-scoped requirements files (one per conda env):

| File | Conda env | Backends covered |
|---|---|---|
| `baseline_core.txt` | `baseline_core` | `trl_deepspeed`, `trl_fsdp`, `deepspeed_arctic`, `megatrain`, `torchtitan` |
| `baseline_megatron.txt` | `baseline_megatron` | `megatron` |

The five core backends share one env because their pip deps are
mutually compatible. Megatron lives alone because
`transformer-engine` pins torch tightly.

Neither file pins `torch` / `torchvision` / `torchaudio` /
`flash-attn*` — those are auto-installed by
`baseline/scripts/install_backend.sh` based on the local CUDA version
detected by `baseline/scripts/detect_cuda.py`. To install:

```bash
baseline/scripts/install_backend.sh --backend trl_fsdp
```

The harness itself is not installed as `flextrain`; it is used directly
from the repo through `PYTHONPATH`.
