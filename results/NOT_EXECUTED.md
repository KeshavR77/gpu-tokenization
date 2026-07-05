# Execution status: NOT RUN ON GPU

This benchmark was **implemented and validated on a non-GPU machine** (Apple Silicon macOS —
no `nvidia-smi`, no CUDA, no `vllm` installed), so the end-to-end run against
`Qwen/Qwen3.5-9B` on the A6000 has **not** been executed here. No `bench.json`, per-mode
`server.log`/`metrics.txt`, or real `summary.md` exist yet — they are produced by running
`python run_benchmark.py` on the A6000 (see `README.md`).

## What *was* validated locally (no GPU needed)

- `run_benchmark.py` compiles and its CLI/dry-run work (`dry_run_preview.txt` in this folder).
- The exact `vllm serve` / `vllm bench serve` commands per mode (see `dry_run_preview.txt`).
- Unit-tested pure logic: Prometheus counter parsing, acceptance-length **τ** computation
  (metrics + log paths), OOM detection, failure-reason extraction, `bench.json` reading, and
  `summary.md` rendering with mixed OK/FAILED statuses and correct speedup.

## To produce real results

On the A6000, after the setup in `README.md`:

```bash
python run_benchmark.py            # runs A, B, C end-to-end
```

This overwrites/creates `results/<mode>/{server.log,bench.json,bench_stdout.log,metrics.txt}`,
`results/run_env.txt`, and the real `results/summary.md` + `summary.json`.

---

## Illustrative summary (SHAPE ONLY — fabricated numbers, not a real run)

The real `summary.md` will look like this (values here are invented to show the format and
the expected-outcome interpretations; **do not cite these numbers**):

```
| Mode | Status | Output tok/s | TPOT (ms) | TTFT (ms) | tau (accept len) | Speedup vs A |
|------|--------|-------------:|----------:|----------:|-----------------:|-------------:|
| A - Baseline (no spec)                      | OK     | 42.10 | 23.70 | 55.20 | 1.000 | 1.00x |
| B - Native MTP                              | FAILED |   -   |   -   |   -   |   -   |   -   |
| C - TLI (Llama draft, heterogeneous vocab)  | OK     | 38.40 | 26.00 | 70.10 | 1.120 | 0.91x |

- A: reference baseline (tau = 1.0 by definition).
- B: FAILED — e.g. "mtp not yet supported for this architecture" (a legitimate finding if
  MTP isn't wired for Qwen3.5's hybrid arch on the installed nightly).
- C: tau near 1 (1.12) -> speculation barely firing; speedup 0.91x (below baseline is a valid
  result: small cross-family Llama<->Qwen vocab intersection).
```
