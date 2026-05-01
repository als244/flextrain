# Parity baselines

Recorded full ``train.py`` runs that we treat as ground truth for end-
to-end loss-curve regression checks. Don't delete these — they're how
we detect "did my engine change move the loss curve?"

## Files

* ``baseline_qwen3_5_9b_full.log`` — Qwen3.5-9B, ``--mode full``,
  5 steps, mathinstruct, lr=3e-5 (train.py default for full).
  Recorded losses: 0.7442, 0.5176, 0.4892, 0.4714, 0.4547.
* ``baseline_qwen3_5_35b_a3b_lora.log`` — Qwen3.5-35B-A3B,
  ``--mode lora``, 5 steps, mathinstruct, lr=1e-4 (train.py default
  for lora). Recorded losses: 0.7432, 0.6866, 0.6452, 0.5855, 0.5407.

Both runs were captured on RTX 3090, master at the time was the
pre-C8/C9 state (commit cf71e81).

## Re-running

The harness ``tests/parity_qwen3_5_9b_35b_5step.py`` runs both back-
to-back and compares per-step losses against the recorded values. The
default tolerance is 5e-3 absolute per step (bf16 noise floor).

```bash
python tests/parity_qwen3_5_9b_35b_5step.py
# or to skip one
python tests/parity_qwen3_5_9b_35b_5step.py --skip-9b
python tests/parity_qwen3_5_9b_35b_5step.py --skip-35b
```

## Memory budget

Both baselines used ``--leeway-gpu-mem-gib 2`` (NOT the train.py
default of 5.0), which is necessary to fit the 9B/35B workloads on a
24 GiB card. The harness sets this automatically.

Settings that match the recorded baselines:

```
--max-seq-len 2048 --max-global-batch-tokens 65536 --steps 5
--max-gpu-mem-gib 22.5 --max-host-mem-gib 110 --leeway-gpu-mem-gib 2
--dataset datasets/mathinstruct.jsonl
```

## When to update the baselines

Don't update lightly. Update only when an INTENTIONAL change shifts
the loss curve (e.g., a new optimizer, a different LR schedule, a
math correctness fix that legitimately produces different numbers).
When you do update, capture the new log to this directory and update
the recorded losses in
``tests/parity_qwen3_5_9b_35b_5step.py:REFERENCE``. Note in the commit
WHY the curve shifted.
