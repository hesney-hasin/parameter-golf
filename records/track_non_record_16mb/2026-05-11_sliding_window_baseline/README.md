# Sliding Window Evaluation — Non-Record Submission

**Author:** Hasnay Hasin (@hesney-hasin)
**Track:** Non-record / Unlimited Compute
**Score:** TBD (to be updated after training run)
**Date:** 2026-05-11

## What I Changed
Applied sliding window evaluation (stride=64) on top of the naive baseline.
Instead of evaluating each sequence independently (context resets to zero each time),
the sliding window lets the model see overlapping context — giving better predictions
and lower bits-per-byte (bpb) score.

## Why This Helps
In the naive baseline, every evaluation sequence starts fresh with no context.
With sliding window (stride=64), each new sequence overlaps with the previous one
by (seq_len - 64) tokens. This means the model has seen most of the context already,
so it predicts more accurately.

## Architecture (Unchanged from Baseline)
- 9 transformer layers
- 512 model dimension
- 1024 vocabulary size (sp1024 tokenizer)
- Tied input/output embeddings

## Hardware
- Prototyping: Kaggle 2×T4
- Final run: Runpod H100 (pending compute grant)

## Results
| Metric | Value |
|--------|-------|
| val_bpb | TBD |
| Model size | TBD |

## How to Reproduce
```bash
RUN_ID=sliding_window_baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 \
records/track_non_record_16mb/2026-05-11_sliding_window_baseline/train_gpt.py
```

## References
- Naive Baseline PR: https://github.com/openai/parameter-golf/pull/1
- Sliding Window Eval inspiration: Matthew Li's submission (1.1925 bpb)
