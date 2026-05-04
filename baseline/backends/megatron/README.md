# Megatron Backend Adapter

`baseline/run_baseline.py --backend megatron` generates a `model_dims.json` from the HuggingFace model config and launches a Megatron Core training script.

Script lookup order:

1. `baseline/backends/megatron/train.py`
2. `orig/baseline/megatron/train.py`

Drop a cleaned-up Megatron script at `baseline/backends/megatron/train.py` when you are ready to stop depending on the old `orig` version.
