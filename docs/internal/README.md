# Internal docs

Working documents for FlexTrain development — running design logs,
in-progress refactor plans, and session notes. Not part of the user-
facing documentation set in `docs/` (see the top-level
[README](../../README.md)'s Documentation table for that).

Production code in `flextrain/` may still link here as the canonical
record for a given design decision; treat the files below as the
source-of-truth for "why is it like that?" questions, not as a
tutorial.

| | |
|---|---|
| [START_HERE.md](START_HERE.md) | entry point + phase ordering for the AdaWS → FlexTrain port |
| [PLAN.md](PLAN.md) | full implementation plan |
| [NOTES.md](NOTES.md) | running decision log + findings |
| [SESSION_NOTES.md](SESSION_NOTES.md) | session-by-session work log |
| [lora_fast_backward.md](lora_fast_backward.md) | LoRA fast-path backward refactor (phased rollout) |
| [multi_chunk_seq_handling.md](multi_chunk_seq_handling.md) | dense + linear-attn multi-chunk mechanism |
| [cross_chunk_lin_attn_plan.md](cross_chunk_lin_attn_plan.md) | FLA state plumbing across chunks |
| [diffusion_transformer.md](diffusion_transformer.md) | DiT support exploration |
