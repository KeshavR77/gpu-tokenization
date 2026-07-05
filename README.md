# Three-Mode Speculative Decoding Benchmark — Qwen3.5-9B

A minimal, config-driven harness that benchmarks three decoding configurations of
`Qwen/Qwen3.5-9B` on a single **NVIDIA A6000 (48 GB)** using vLLM's built-in
`vllm bench serve` against the ShareGPT dataset, and produces a throughput / latency /
acceptance-length comparison.

Speculative decoding is **output-distribution-preserving** — at temperature 0 all three
modes emit *identical* tokens, so we measure **speed only**. Mode A is the baseline the two
speculative modes are measured against.

## The three modes

| Mode | Description | Speculative config |
|---|---|---|
| **A — Baseline** | No speculative decoding | *(none)* |
| **B — Native MTP** | Qwen3.5's built-in multi-token-prediction head (same tokenizer) | `{"method":"mtp","num_speculative_tokens":3}` |
| **C — TLI** | Cross-tokenizer speculation with a Llama draft | `{"method":"draft_model","model":"meta-llama/Llama-3.2-1B-Instruct","num_speculative_tokens":3,"use_heterogeneous_vocab":true}` |

Only the speculative config changes between modes. Everything else (target model, dataset,
seed, `num_speculative_tokens=3`, greedy sampling) is identical — that is what makes the
comparison fair.

> **TLI API note:** cross-tokenizer speculation is `method:"draft_model"` **plus**
> `use_heterogeneous_vocab:true` (which builds a token-level intersection of the two
> vocabularies at init and constrains the draft to shared tokens). It is *not*
> `method:"universal_draft"`. `use_heterogeneous_vocab` currently supports **greedy draft
> sampling only** → every mode runs at **temperature 0**.

## Requirements

- One NVIDIA A6000 (48 GB) or comparable. `Qwen/Qwen3.5-9B` is a multimodal, hybrid
  (Gated-DeltaNet + MoE) model whose architecture is **only in vLLM nightly/main**, not a
  stable release.
- `--language-model-only` is passed so the vision tower is skipped (text-only benchmark).

## Setup

```bash
# vLLM nightly is REQUIRED for Qwen3.5 (hybrid arch is not in stable releases)
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

# Llama 3.2 (mode C draft) is a gated repo; Qwen3.5 is apache-2.0
huggingface-cli login

# ShareGPT dataset (into this directory)
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Run

```bash
# All three modes, task defaults (500 prompts, concurrency 1, seed 42):
python run_benchmark.py

# Preview the exact serve/bench commands without launching anything:
python run_benchmark.py --dry-run

# A subset, or tweak the knobs:
python run_benchmark.py --modes A,C --num-prompts 200 --concurrency 1 --seed 42
```

Useful flags (all default to the task's fixed parameters):

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-9B` | target model |
| `--draft-model` | `meta-llama/Llama-3.2-1B-Instruct` | mode-C draft |
| `--dataset-path` | `./ShareGPT_V3_unfiltered_cleaned_split.json` | ShareGPT json |
| `--num-prompts` | `500` | prompts sent to `vllm bench serve` |
| `--concurrency` | `1` | `--max-concurrency` (batch-1 latency regime) |
| `--seed` | `42` | seed (server + client) |
| `--max-model-len` | `8192` | context cap (native 262K would OOM) |
| `--gpu-mem-util` | `0.9` | `--gpu-memory-utilization` |
| `--num-spec-tokens` | `3` | `num_speculative_tokens` for **both** spec modes |
| `--modes` | `A,B,C` | subset to run |
| `--startup-timeout` | `1800` | seconds to wait for `/health` (nightly load is slow) |
| `--gpu-id` | `0` | GPU index (sets `CUDA_VISIBLE_DEVICES`) |

## What the harness does (per mode, one server at a time)

1. **Launch** `vllm serve` as a subprocess in its own process group; stdout+stderr →
   `results/<mode>/server.log`.
2. **Wait** by polling `http://host:port/health` until `200` (up to `--startup-timeout`).
   If the process exits early or never becomes ready → kill the process group, mark the mode
   **FAILED**, capture the reason from the log, and continue to the next mode.
3. **Benchmark** with `vllm bench serve --save-result` → `results/<mode>/bench.json`.
4. **Scrape** `http://host:port/metrics` → `results/<mode>/metrics.txt` (done *before*
   teardown, while the counters are still live).
5. **Tear down** (SIGINT → SIGTERM → SIGKILL on the group) and **poll `nvidia-smi`** until
   GPU memory returns to the pre-mode baseline before starting the next mode.

Isolation is deliberate: one server at a time, no shared state.

### OOM handling
If a server fails to start and the log shows an out-of-memory error, the harness retries the
mode **once** at reduced footprint (`gpu_mem_util -= 0.1`, `max_model_len //= 2`, floored at
`4096`) and records the change in `results/summary.md`. It never silently drops a mode.

### Fallback (offline)
`vllm bench serve` is preferred because it reports per-request TPOT/ITL. If server-lifecycle
management is too fragile in your environment, run the offline in-process path directly (one
mode shown — mode A drops `--speculative-config`, mode B swaps in the MTP config):

```bash
vllm bench throughput \
  --model Qwen/Qwen3.5-9B --backend vllm \
  --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 --seed 42 \
  --language-model-only --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --output-json results/C_tli/bench_offline.json \
  --speculative-config '{"method":"draft_model","model":"meta-llama/Llama-3.2-1B-Instruct","num_speculative_tokens":3,"use_heterogeneous_vocab":true}'
```

Caveats: this runs in-process (no server), does **not** honor `--max-concurrency 1` (it is a
throughput-oriented path), and does not report TPOT/ITL. Read τ from its stdout/log
(acceptance-length line) since there is no `/metrics` endpoint to scrape.

## Metrics (how each is computed)

Written to `results/summary.md` (also printed) and `results/summary.json`.

- **Output throughput (tok/s)** — `output_throughput` from `bench.json`.
- **TPOT (ms)** — `mean_tpot_ms` (falls back to `median_tpot_ms`) from `bench.json`.
- **TTFT (ms)** — `mean_ttft_ms` (falls back to `median_ttft_ms`) from `bench.json`.
- **Acceptance length τ** (spec modes only) — mean tokens accepted per decode step. This is a
  **server-side** metric, not in `bench.json`. Computed, in priority order, from:
  1. Prometheus counters in `/metrics`: `τ ≈ 1 + accepted / num_drafts`, using
     `vllm:spec_decode_num_accepted_tokens_total` and `vllm:spec_decode_num_drafts_total`
     (or `.../num_draft_tokens_total ÷ num_spec_tokens` when `num_drafts` is absent).
  2. The server log's periodic "acceptance length" line.

  Baseline has no speculation → **τ = 1.0** by definition. Counter names are matched loosely
  so minor version renames still resolve; verify against your installed version if τ shows as
  `-`.
- **Speedup** — mode output throughput ÷ Mode A (baseline) throughput.

### Greedy / temperature 0
Required by TLI and gives determinism. vLLM's serving benchmark issues requests at
temperature 0 (greedy) **by default**; the harness *also* passes `--temperature 0` to
`vllm bench serve` when the installed version exposes that flag. Which path was used is
recorded in the summary header and in `results/run_env.txt`.

## Expected outcomes (these are findings, not bugs)

- **Mode B (MTP) may fail to start** if MTP isn't yet wired for Qwen3.5's hybrid architecture
  on the installed nightly. It is caught, marked FAILED with the log reason, and reported.
- **Mode C (TLI) may show low τ or a slowdown** (speedup < 1.0): the Llama↔Qwen vocabulary
  intersection is small and cross-family. A below-baseline result is recorded as-is.
- **OOM** → automatic one-shot retry at reduced footprint, noted in the summary.

## Outputs

```
results/
  run_env.txt            # nvidia-smi + vllm --version + args, captured at start
  <mode>/server.log      # full server stdout+stderr
  <mode>/bench.json      # vllm bench serve --save-result output
  <mode>/bench_stdout.log
  <mode>/metrics.txt     # scraped Prometheus /metrics
  summary.md             # comparison table + per-mode interpretation
  summary.json           # machine-readable results
```

## Execution status

This harness was **developed and validated on a non-GPU machine** (Apple Silicon macOS, no
`nvidia-smi`/`vllm`). Syntax, the command builders, flag-probing, `/metrics` + log parsing,
τ computation, OOM/failure detection, and summary rendering are unit-tested; the end-to-end
GPU run must be executed on the A6000. See `results/NOT_EXECUTED.md`.
