# Speculative Decoding Benchmark - Qwen3-8B (Baseline vs TLI)

- Generated: 2026-07-11 05:33:13 UTC
- Host: de8be2164782
- GPU: NVIDIA RTX A6000
- vLLM: 0.25.1.dev34+g092387963
- Model: Qwen/Qwen3-8B   Draft (mode C): meta-llama/Llama-3.2-1B-Instruct
- Fixed: max-model-len=8192, gpu-mem-util=0.9, concurrency=1, num-prompts=500, seed=42, num_spec_tokens=3
- Sampling: greedy / temperature 0 - no `--temperature` flag in installed `vllm bench serve`; relying on vLLM's benchmark default of temperature 0 (greedy). Verify in server.log per-request sampling params.

| Mode | Status | Output tok/s | TPOT (ms) | TTFT (ms) | tau (accept len) | Speedup vs A |
|------|--------|-------------:|----------:|----------:|-----------------:|-------------:|
| A - Baseline (no spec) | OK | 41.10 | 24.13 | 62.72 | 1.000 | 1.00x |
| C - TLI (Llama draft, heterogeneous vocab) | OK | 50.23 | 21.23 | 84.32 | 2.359 | 1.22x |

## Interpretation
- **A - Baseline (no spec)**: reference baseline (tau = 1.0 by definition).
- **C - TLI (Llama draft, heterogeneous vocab)**: high tau (2.36) -> real speedup (1.22x). Speculation is firing.

_Acceptance length tau = mean tokens accepted per decode step (1 + accepted_tokens/num_drafts from `vllm:spec_decode_*` counters, or the server log's reported acceptance length). Baseline has no speculation -> tau = 1.0. Speculative decoding is lossless, so both modes emit identical tokens at temperature 0; only speed differs._
