# Dataset format and training-data specification

FlexTrain consumes training data as a stream of **sequences**. A
sequence is a small bag of three tensors:

```python
seq.tokens   # (T,) int64 — the input token IDs
seq.targets  # (T,) int64 — for each prediction position, the target token
             #              ID, or -100 to ignore that position in loss
seq.seq_id   # int       — opaque identifier (informational only)
```

There is no Dataset / DataLoader abstraction layered on top — the engine
takes a plain Python list of sequences per training step. You build that
list however you like.

## The minimum sequence type

```python
class _Seq:
    tokens: torch.Tensor   # (T,) int64
    targets: torch.Tensor  # (T,) int64, with -100 for ignored positions
    seq_id: int = 0
```

There's a stub in `flextrain/bench/parity.py:_Seq` that constructs
`targets = torch.roll(tokens, -1)` (next-token prediction with the last
position rolling around). For SFT or anything else where you need to
mask part of the prediction, set `targets[i] = -100` for positions you
want excluded from the loss.

You can use any class — duck typing on `.tokens`, `.targets`, and
`.seq_id` is all the engine needs.

## A training step

```python
loss = _flextrain_step(am, seqs)
```

* `seqs: list[Seq]` — variable-length sequences. The engine packs them
  into chunks (per `chunk.q_seq_lens` / `chunk.q_seq_offsets`), routes
  through the scheduler, and runs forward + backward + optimizer step.
* Returns: scalar token-averaged cross-entropy over **active** positions
  (those with `targets[i] != -100`).

Internally:

1. The scheduler (`flextrain/core/schedule.py`) packs sequences into
   round / chunk buckets that fit the working-set budget.
2. The engine accumulates gradients across chunks within a round, then
   calls the optimizer once per round.
3. Loss scaling: per-token cross-entropy is summed across chunks and
   divided by the global active-token count, matching what a single big
   forward + backward would yield.

## SFT prompt masking

For supervised fine-tuning, mask the loss over the prompt portion:

```python
prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
response_ids = tok.encode(response_text, add_special_tokens=False)
total_ids = prompt_ids + response_ids

tokens = torch.tensor(total_ids, dtype=torch.int64)
targets = torch.roll(tokens, -1)
targets[: len(prompt_ids)] = -100   # ignore prompt-position predictions
targets[-1] = -100                  # roll wraparound — also -100

seq = _Seq(tokens)
seq.targets = targets
```

This is exactly what `tests/test_llama32_1b_parity.py:_pull_step_batches`
does for the MathInstruct dataset.

## Iterating real corpora

There's a couple of starting points you can copy, depending on the
data format you have:

### Bundled local JSON SFT source

For the common "instruction/output JSON file" case, the CLI now has a
built-in adapter. Point `io.data.json_sft_path` at a local `.json` or
`.jsonl` file and set `io.tokenizer` to the tokenizer you want to use.

```yaml
io:
  tokenizer: gpt2
  data:
    json_sft_path: path/to/data.json
    json_sft_prompt_field: instruction
    json_sft_response_field: output
    json_sft_input_field: input
```

Each record is tokenized as prompt + response, and the prompt tokens are
masked out of the loss automatically.

### Auto-materialized datasets via `train.py`

The top-level `train.py` entrypoint keeps a single `--dataset` flag.
If the path exists locally, it uses that JSON / JSONL file directly.
If it does not exist, FlexTrain first tries to materialize it into a
local JSONL file and then trains from that file:

```bash
python train.py \
  --model models/Llama-3.1-8B \
  --mode lora \
  --max-seq-len 1024 \
  --max-global-batch-tokens 1024 \
  --dataset open-r1/OpenR1-Math-220k
```

Today that materialization path is aimed at supervised fine-tuning
datasets. It recognizes common schemas such as:

* `instruction` / `output` (+ optional `input`)
* `prompt` / `completion`
* `prompt` / `response`
* `question` / `answer` (+ optional `context`)
* chat-style `messages` / `conversations`

### Pre-tokenized binary shards (FineWeb-style)

`flextrain/bench/parity.py:FineWebDocStream` yields sequences from
GPT-2-tokenized FineWeb `.bin` shards (256 int32 header + uint16
tokens, EOT = 50256). For Llama / Qwen training you'd typically detok
+ retok with the target tokenizer; otherwise just feed the GPT-2
tokens directly into the engine — no tokenizer is required at training
time.

### JSON instruction datasets (MathInstruct-style)

`tests/test_llama32_1b_parity.py:_pull_step_batches` shows the pattern:
load a list of `{"instruction": ..., "output": ...}` records, encode
prompt + response with HF `AutoTokenizer`, pack into sequences with
SFT prompt masking. ~50 lines.

### Custom sources

The engine takes any object with `.tokens` / `.targets` / `.seq_id`.
If you have a HuggingFace `datasets`-style stream, write a thin adapter
that yields `_Seq` objects in your chosen batch granularity.

## Batch building

Sequences in one batch have **independent positions**. The flash-attn
varlen path handles variable-length packing — you don't have to pad.

A batch's total token count should approximate `target_round_tokens`
in your `WorkingSetConfig` for predictable memory use. The
`_pull_step_batches` helper builds batches that aim for a target
token total per step:

```python
def _build_seq() -> _Seq | None: ...

step_batches: list[list[_Seq]] = []
for _ in range(n_steps):
    batch, total = [], 0
    while total < target_tokens_per_step:
        seq = _build_seq()
        batch.append(seq)
        total += len(seq)
    step_batches.append(batch)
```

## What if I have padding tokens?

You typically don't pad in FlexTrain — the engine is designed for
variable-length packing. If your data already has `<pad>` tokens you
want to ignore, set `targets[i] = -100` at those positions. The pad
token will still be embedded and consume compute, but won't contribute
to the loss.

## Multi-modality, image tokens, etc.

The engine treats `tokens` as opaque int64 indices into the embedding
table. Image / audio tokens are fine as long as the embedding table
covers them. Cross-modal layers / non-text architectures are not
supported yet; raise an issue or implement a custom block.

## Determinism

For reproducibility:
* Set `torch.manual_seed(...)` before constructing the model and before
  iterating data.
* The engine itself is deterministic for a fixed working-set config.
  Save-level changes (e.g. forcing more activation recompute) are
  bit-identical at the loss level — the working-set planner only
  changes WHERE intermediates live, not what's computed.
* Naive vs offloaded with the same optimizer is also bit-identical:
  parameter / gradient / optimizer-state offload only changes the
  storage location, not the math.
