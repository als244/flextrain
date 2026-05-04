"""FSDP2 + TRL SFTTrainer baseline (HF model parallelism via accelerate's FSDP plugin).

Designed as the apples-to-apples FSDP counterpart to ``trl_deepspeed``: identical
training loop (TRL SFTTrainer over synthetic random tokens), only the parallelism
plugin differs. Used for HF model families that TorchTitan's registry does not
cover (e.g. Qwen3.5, Qwen3.5-MoE).
"""
