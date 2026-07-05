#!/usr/bin/env python3
"""
Three-mode speculative-decoding benchmark for Qwen/Qwen3.5-9B on a single A6000.

For each mode (A baseline / B native-MTP / C TLI cross-tokenizer draft) the harness:
  1. Launches a `vllm serve` subprocess (stdout+stderr -> results/<mode>/server.log)
  2. Polls /health until ready (generous timeout); on failure: kill, mark FAILED, continue
  3. Runs `vllm bench serve` against it (--save-result -> results/<mode>/bench.json)
  4. Scrapes /metrics -> results/<mode>/metrics.txt
  5. Tears down the server and waits until GPU memory is freed before the next mode.

Then it writes results/summary.md (and summary.json) comparing throughput / TPOT / TTFT /
acceptance-length tau / speedup.

Speculative decoding is output-distribution-preserving, so at temperature 0 all three modes
emit *identical* tokens; we are measuring speed only.

Robustness over cleverness: one server at a time, each mode fully isolated, no shared state.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------------------
# Mode definitions. Only the speculative config changes across modes; everything else is
# fixed (see build_serve_cmd). `num_speculative_tokens` is injected from CLI so B and C
# always match -- that is what makes the comparison fair.
# --------------------------------------------------------------------------------------
def build_modes(num_spec_tokens: int, draft_model: str):
    return [
        {
            "name": "A_baseline",
            "label": "A - Baseline (no spec)",
            "spec": False,
            "speculative_config": None,
        },
        {
            "name": "B_mtp",
            "label": "B - Native MTP",
            "spec": True,
            # Qwen3.5 exposes a built-in multi-token-prediction head (same tokenizer).
            # vLLM infers the MTP module from the model; "mtp" is the documented method
            # string. If the installed nightly rejects it, the failure is captured and
            # reported (MTP may not yet be wired for the hybrid arch) -- a valid finding.
            "speculative_config": {
                "method": "mtp",
                "num_speculative_tokens": num_spec_tokens,
            },
        },
        {
            "name": "C_tli",
            "label": "C - TLI (Llama draft, heterogeneous vocab)",
            "spec": True,
            # Token-Level-Intersection cross-tokenizer speculation: a Llama 1B draft with a
            # token-level intersection of the two vocabularies. API is method=draft_model
            # PLUS use_heterogeneous_vocab=true (NOT method="universal_draft").
            # use_heterogeneous_vocab supports greedy draft sampling only -> temperature 0.
            "speculative_config": {
                "method": "draft_model",
                "model": draft_model,
                "num_speculative_tokens": num_spec_tokens,
                "use_heterogeneous_vocab": True,
            },
        },
    ]


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    print(f"[{now_utc()}] {msg}", flush=True)


def run_capture(cmd, timeout=60):
    """Run a command, return (returncode, stdout+stderr). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout running: {' '.join(map(str, cmd))}"
    except Exception as e:  # pragma: no cover - defensive
        return 1, f"error running {cmd}: {e}"


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


# --------------------------------------------------------------------------------------
# GPU helpers (all guarded -- harness stays usable on machines without nvidia-smi)
# --------------------------------------------------------------------------------------
HAVE_NVIDIA_SMI = shutil.which("nvidia-smi") is not None


def gpu_used_mb(gpu_id: int):
    if not HAVE_NVIDIA_SMI:
        return None
    rc, out = run_capture(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits", "-i", str(gpu_id)],
        timeout=15,
    )
    if rc != 0:
        return None
    try:
        return int(out.strip().splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


def wait_for_gpu_free(baseline_mb, gpu_id: int, timeout: int = 180, delta_mb: int = 800):
    """Wait until GPU used memory returns to (baseline + delta). Returns a status string."""
    if not HAVE_NVIDIA_SMI or baseline_mb is None:
        return "gpu-check-skipped (no nvidia-smi / no baseline)"
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        used = gpu_used_mb(gpu_id)
        last = used
        if used is not None and used <= baseline_mb + delta_mb:
            return f"gpu freed (used={used}MB, baseline={baseline_mb}MB)"
        time.sleep(3)
    return f"gpu-free-timeout after {timeout}s (used={last}MB, baseline={baseline_mb}MB)"


# --------------------------------------------------------------------------------------
# Prometheus / log parsing for acceptance length tau
# --------------------------------------------------------------------------------------
def metric_value(text: str, metric_name: str):
    """Sum all samples of a Prometheus counter across label sets. None if absent."""
    if not text:
        return None
    total = 0.0
    found = False
    pat = re.compile(rf"^{re.escape(metric_name)}(\{{[^}}]*\}})?\s+([0-9.eE+\-]+)\s*$")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = pat.match(line)
        if m:
            try:
                total += float(m.group(2))
                found = True
            except ValueError:
                pass
    return total if found else None


def compute_tau(mode, metrics_text, log_text, num_spec_tokens):
    """
    Acceptance length tau = mean tokens accepted per decode step.
      tau ~= 1 + accepted_tokens / num_drafts
    Baseline (no speculation) is 1.0 by definition. Returns (tau_or_None, source_note).
    """
    if not mode["spec"]:
        return 1.0, "baseline (no speculation)"

    # (a) Prometheus counters -- most reliable. Names verified against vLLM V1 spec-decode
    #     metrics; matched loosely so minor renames still resolve.
    if metrics_text:
        accepted = metric_value(metrics_text, "vllm:spec_decode_num_accepted_tokens_total")
        drafts = metric_value(metrics_text, "vllm:spec_decode_num_drafts_total")
        draft_tokens = metric_value(metrics_text, "vllm:spec_decode_num_draft_tokens_total")
        if accepted is not None and drafts and drafts > 0:
            return 1.0 + accepted / drafts, "metrics: 1 + accepted/num_drafts"
        if accepted is not None and draft_tokens and draft_tokens > 0 and num_spec_tokens:
            est_drafts = draft_tokens / num_spec_tokens
            if est_drafts > 0:
                return (1.0 + accepted / est_drafts,
                        "metrics: 1 + accepted/(draft_tokens/k)")

    # (b) Server log -- vLLM periodically prints an acceptance-length line.
    if log_text:
        vals = re.findall(r"acceptance length[:=\s]+([0-9]+\.?[0-9]*)",
                          log_text, re.IGNORECASE)
        if vals:
            return float(vals[-1]), "server log: reported acceptance length"

    return None, "unavailable (no spec-decode metrics found)"


# --------------------------------------------------------------------------------------
# Failure-reason extraction from a server log
# --------------------------------------------------------------------------------------
_INTERESTING = re.compile(
    r"(error|exception|not supported|unsupported|out of memory|\boom\b|"
    r"notimplemented|invalid|traceback|assert|failed|raise )",
    re.IGNORECASE,
)


def detect_reason(log_text: str) -> str:
    if not log_text:
        return "no server output captured"
    candidates = [l.strip() for l in log_text.splitlines()
                  if l.strip() and _INTERESTING.search(l)]
    if candidates:
        return candidates[-1][:400]
    return "server did not become ready (no explicit error; see server.log)"


def is_oom(log_text: str) -> bool:
    return bool(re.search(r"out of memory|CUDA out of memory|\bOOM\b", log_text or "",
                          re.IGNORECASE))


def tail(text: str, n: int = 40) -> str:
    return "\n".join((text or "").splitlines()[-n:])


# --------------------------------------------------------------------------------------
# vllm bench serve flag probing (installed version may or may not expose sampling flags)
# --------------------------------------------------------------------------------------
_BENCH_HELP_CACHE = None


def bench_help_text():
    global _BENCH_HELP_CACHE
    if _BENCH_HELP_CACHE is None:
        _, out = run_capture(["vllm", "bench", "serve", "--help"], timeout=60)
        _BENCH_HELP_CACHE = out or ""
    return _BENCH_HELP_CACHE


def bench_supports(flag: str) -> bool:
    return flag in bench_help_text()


# --------------------------------------------------------------------------------------
# Command builders
# --------------------------------------------------------------------------------------
def build_serve_cmd(mode, args, gpu_mem_util, max_model_len):
    cmd = [
        "vllm", "serve", args.model,
        "--language-model-only",           # skip the vision tower (text-only benchmark)
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--seed", str(args.seed),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if mode["speculative_config"] is not None:
        cmd += ["--speculative-config",
                json.dumps(mode["speculative_config"], separators=(",", ":"))]
    return cmd


def build_bench_cmd(mode, args, result_file):
    result_file = Path(result_file)
    cmd = [
        "vllm", "bench", "serve",
        "--model", args.model,
        "--dataset-name", "sharegpt",
        "--dataset-path", args.dataset_path,
        "--num-prompts", str(args.num_prompts),
        "--max-concurrency", str(args.concurrency),
        "--seed", str(args.seed),
        "--host", args.host,
        "--port", str(args.port),
        "--save-result",
    ]
    # Some versions join --result-dir + --result-filename; others take a full path in
    # --result-filename. Use --result-dir when available so bench.json lands where we read it.
    if bench_supports("--result-dir"):
        cmd += ["--result-dir", str(result_file.parent),
                "--result-filename", result_file.name]
    else:
        cmd += ["--result-filename", str(result_file)]
    # Greedy / temperature 0 is required by TLI and gives determinism. vLLM's serving
    # benchmark issues requests at temperature 0 by default; we additionally pass the flag
    # when the installed version exposes it. (Recorded in the run env / summary.)
    if bench_supports("--temperature"):
        cmd += ["--temperature", "0"]
    return cmd


def greedy_enforcement_note():
    if bench_supports("--temperature"):
        return "temperature 0 passed explicitly via `--temperature 0` to `vllm bench serve`"
    return ("no `--temperature` flag in installed `vllm bench serve`; relying on vLLM's "
            "benchmark default of temperature 0 (greedy). Verify in server.log per-request "
            "sampling params.")


# --------------------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------------------
def http_get(url: str, timeout: float = 5.0):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def wait_for_health(args, proc, startup_timeout: int):
    url = f"http://{args.host}:{args.port}/health"
    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"server process exited early (code {proc.returncode})"
        try:
            status, _ = http_get(url, timeout=5)
            if status == 200:
                return True, "ready"
        except Exception:
            pass  # not up yet
        time.sleep(3)
    return False, f"health check timed out after {startup_timeout}s"


def scrape_metrics(args, out_path: Path):
    url = f"http://{args.host}:{args.port}/metrics"
    try:
        status, body = http_get(url, timeout=15)
        if status == 200:
            out_path.write_text(body)
            return body
    except Exception as e:
        log(f"  metrics scrape failed: {e}")
    return None


# --------------------------------------------------------------------------------------
# Server lifecycle
# --------------------------------------------------------------------------------------
def start_server(cmd, log_path: Path, env):
    log_f = open(log_path, "w")
    log_f.write(f"# launched: {now_utc()}\n# cmd: {' '.join(map(str, cmd))}\n\n")
    log_f.flush()
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,  # own process group -> clean group kill
    )
    return proc, log_f


def stop_server(proc, log_f):
    """SIGINT -> SIGTERM -> SIGKILL on the whole process group."""
    if proc.poll() is None:
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None
        for sig, wait_s in ((signal.SIGINT, 20), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
            if proc.poll() is not None:
                break
            try:
                if pgid is not None:
                    os.killpg(pgid, sig)
                else:
                    proc.send_signal(sig)
            except ProcessLookupError:
                break
            end = time.time() + wait_s
            while time.time() < end and proc.poll() is None:
                time.sleep(0.5)
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# bench.json reader
# --------------------------------------------------------------------------------------
def read_bench(path: Path):
    data = json.loads(path.read_text())

    def g(*keys):
        for k in keys:
            v = data.get(k)
            if v is not None:
                return v
        return None

    return {
        "output_throughput": g("output_throughput"),
        "total_token_throughput": g("total_token_throughput"),
        "request_throughput": g("request_throughput"),
        "tpot_ms": g("mean_tpot_ms", "median_tpot_ms"),
        "ttft_ms": g("mean_ttft_ms", "median_ttft_ms"),
        "itl_ms": g("mean_itl_ms", "median_itl_ms"),
        "completed": g("completed"),
        "total_output_tokens": g("total_output_tokens"),
        "duration": g("duration"),
    }


# --------------------------------------------------------------------------------------
# Per-mode run (with one OOM-triggered retry at reduced memory footprint)
# --------------------------------------------------------------------------------------
def run_mode(mode, args, env):
    outdir = Path(args.results_dir) / mode["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": mode["name"], "label": mode["label"], "spec": mode["spec"],
        "status": "FAILED", "reason": "", "output_throughput": None,
        "tpot_ms": None, "ttft_ms": None, "tau": None, "tau_source": None,
        "speculative_config": mode["speculative_config"], "notes": [],
    }

    gpu_mem_util = args.gpu_mem_util
    max_model_len = args.max_model_len

    for attempt in (1, 2):
        server_log = outdir / "server.log"
        baseline_mb = gpu_used_mb(args.gpu_id)
        cmd = build_serve_cmd(mode, args, gpu_mem_util, max_model_len)
        log(f"[{mode['name']}] attempt {attempt}: launching server "
            f"(gpu_mem_util={gpu_mem_util}, max_model_len={max_model_len})")
        log(f"[{mode['name']}]   $ {' '.join(map(str, cmd))}")

        proc, log_f = start_server(cmd, server_log, env)
        try:
            ok, reason = wait_for_health(args, proc, args.startup_timeout)
            if not ok:
                stop_server(proc, log_f)
                log_text = server_log.read_text(errors="replace") if server_log.exists() else ""
                detected = detect_reason(log_text)
                if is_oom(log_text) and attempt == 1:
                    old = (gpu_mem_util, max_model_len)
                    gpu_mem_util = round(max(0.5, gpu_mem_util - 0.1), 2)
                    max_model_len = max(4096, max_model_len // 2)
                    note = (f"OOM on attempt 1 ({old[0]},{old[1]}); retrying at "
                            f"gpu_mem_util={gpu_mem_util}, max_model_len={max_model_len}")
                    log(f"[{mode['name']}] {note}")
                    result["notes"].append(note)
                    wait_for_gpu_free(baseline_mb, args.gpu_id, args.gpu_free_timeout)
                    continue
                result["status"] = "FAILED"
                result["reason"] = f"{reason}: {detected}"
                log(f"[{mode['name']}] FAILED to start: {result['reason']}")
                wait_for_gpu_free(baseline_mb, args.gpu_id, args.gpu_free_timeout)
                return result

            # ---- server is up: run the benchmark ----
            log(f"[{mode['name']}] server ready; running `vllm bench serve` "
                f"({args.num_prompts} prompts, concurrency {args.concurrency})")
            bench_json = outdir / "bench.json"
            bench_cmd = build_bench_cmd(mode, args, str(bench_json))
            bench_stdout = outdir / "bench_stdout.log"
            with open(bench_stdout, "w") as bf:
                bf.write(f"# cmd: {' '.join(map(str, bench_cmd))}\n\n")
                bf.flush()
                bench_rc = subprocess.run(
                    bench_cmd, stdout=bf, stderr=subprocess.STDOUT,
                    env=env, timeout=args.bench_timeout,
                ).returncode

            # ---- scrape /metrics BEFORE teardown ----
            metrics_text = scrape_metrics(args, outdir / "metrics.txt")

            # ---- teardown + confirm GPU freed ----
            stop_server(proc, log_f)
            free_status = wait_for_gpu_free(baseline_mb, args.gpu_id, args.gpu_free_timeout)
            log(f"[{mode['name']}] teardown: {free_status}")
            result["notes"].append(free_status)

            log_text = server_log.read_text(errors="replace") if server_log.exists() else ""

            # Locate bench.json (fallback: newest *.json the run just wrote under outdir/cwd).
            if not bench_json.exists():
                cands = sorted(
                    [p for p in list(outdir.glob("*.json")) + list(Path(".").glob("*.json"))
                     if p.name not in ("summary.json",)],
                    key=lambda p: p.stat().st_mtime, reverse=True)
                if cands:
                    bench_json = cands[0]
                    result["notes"].append(f"bench result read from {bench_json}")

            if bench_rc != 0 or not bench_json.exists():
                result["status"] = "FAILED"
                result["reason"] = (f"bench serve exited rc={bench_rc}; "
                                    f"see {bench_stdout.name}")
                return result

            b = read_bench(bench_json)
            result["output_throughput"] = b["output_throughput"]
            result["tpot_ms"] = b["tpot_ms"]
            result["ttft_ms"] = b["ttft_ms"]
            result["completed"] = b["completed"]
            tau, tau_src = compute_tau(mode, metrics_text, log_text, args.num_spec_tokens)
            result["tau"] = tau
            result["tau_source"] = tau_src
            result["status"] = "OK"
            log(f"[{mode['name']}] OK: {b['output_throughput']} tok/s, "
                f"TPOT {b['tpot_ms']} ms, tau {tau}")
            return result

        except subprocess.TimeoutExpired:
            stop_server(proc, log_f)
            wait_for_gpu_free(baseline_mb, args.gpu_id, args.gpu_free_timeout)
            result["status"] = "FAILED"
            result["reason"] = f"benchmark timed out after {args.bench_timeout}s"
            return result
        except Exception as e:  # pragma: no cover - defensive
            stop_server(proc, log_f)
            wait_for_gpu_free(baseline_mb, args.gpu_id, args.gpu_free_timeout)
            result["status"] = "FAILED"
            result["reason"] = f"unexpected error: {e}"
            return result

    return result


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------
def fmt(v, spec="{:.2f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "-"


def build_summary(results, args, env_info):
    baseline = next((r for r in results if r["mode"] == "A_baseline"), None)
    base_tp = baseline["output_throughput"] if baseline and baseline["status"] == "OK" else None

    for r in results:
        tp = r["output_throughput"]
        if r["status"] == "OK" and base_tp and isinstance(tp, (int, float)):
            r["speedup"] = tp / base_tp
        else:
            r["speedup"] = None

    lines = []
    lines.append("# Speculative Decoding Benchmark - Qwen3.5-9B\n")
    lines.append(f"- Generated: {now_utc()}")
    lines.append(f"- Host: {env_info['hostname']}")
    lines.append(f"- GPU: {env_info['gpu_name']}")
    lines.append(f"- vLLM: {env_info['vllm_version']}")
    lines.append(f"- Model: {args.model}   Draft (mode C): {args.draft_model}")
    lines.append(f"- Fixed: max-model-len={args.max_model_len}, gpu-mem-util={args.gpu_mem_util}, "
                 f"concurrency={args.concurrency}, num-prompts={args.num_prompts}, "
                 f"seed={args.seed}, num_spec_tokens={args.num_spec_tokens}")
    lines.append(f"- Sampling: greedy / temperature 0 - {greedy_enforcement_note()}")
    lines.append("")
    lines.append("| Mode | Status | Output tok/s | TPOT (ms) | TTFT (ms) | tau (accept len) | Speedup vs A |")
    lines.append("|------|--------|-------------:|----------:|----------:|-----------------:|-------------:|")
    for r in results:
        speedup = f"{r['speedup']:.2f}x" if isinstance(r.get("speedup"), (int, float)) else "-"
        tau = fmt(r["tau"], "{:.3f}") if r["tau"] is not None else "-"
        lines.append(
            f"| {r['label']} | {r['status']} | {fmt(r['output_throughput'])} | "
            f"{fmt(r['tpot_ms'])} | {fmt(r['ttft_ms'])} | {tau} | {speedup} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    for r in results:
        if r["status"] != "OK":
            lines.append(f"- **{r['label']}**: {r['status']} - {r['reason']}")
            continue
        if not r["spec"]:
            lines.append(f"- **{r['label']}**: reference baseline (tau = 1.0 by definition).")
            continue
        tau = r["tau"]
        sp = r.get("speedup")
        if tau is None:
            lines.append(f"- **{r['label']}**: acceptance length unavailable "
                         f"({r['tau_source']}); speedup "
                         f"{('%.2fx' % sp) if sp else 'n/a'}.")
        elif tau >= 2.0 and sp and sp > 1.0:
            lines.append(f"- **{r['label']}**: high tau ({tau:.2f}) -> real speedup "
                         f"({sp:.2f}x). Speculation is firing.")
        elif tau <= 1.3:
            lines.append(f"- **{r['label']}**: tau near 1 ({tau:.2f}) -> speculation barely "
                         f"firing; speedup {('%.2fx' % sp) if sp else 'n/a'}"
                         + (" (below baseline is a valid result: small cross-family vocab "
                            "intersection)." if r["mode"] == "C_tli" and sp and sp < 1
                            else "."))
        else:
            lines.append(f"- **{r['label']}**: tau {tau:.2f}, speedup "
                         f"{('%.2fx' % sp) if sp else 'n/a'}.")
        if r["notes"]:
            for n in r["notes"]:
                if "OOM" in n:
                    lines.append(f"    - note: {n}")
    lines.append("")
    lines.append("_Acceptance length tau = mean tokens accepted per decode step "
                 "(1 + accepted_tokens/num_drafts from `vllm:spec_decode_*` counters, or the "
                 "server log's reported acceptance length). Baseline has no speculation -> "
                 "tau = 1.0. Speculative decoding is lossless, so all modes emit identical "
                 "tokens at temperature 0; only speed differs._")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Environment recording
# --------------------------------------------------------------------------------------
def record_env(args):
    info = {"hostname": socket.gethostname(), "gpu_name": "unknown",
            "vllm_version": "unknown"}
    lines = [f"# Run environment - {now_utc()}", f"host: {info['hostname']}",
             f"args: {json.dumps(vars(args))}", ""]

    rc, out = run_capture(["vllm", "--version"], timeout=60)
    info["vllm_version"] = out.strip().splitlines()[-1] if rc == 0 and out.strip() else "unknown"
    lines += ["## vllm --version", out.strip(), ""]

    if HAVE_NVIDIA_SMI:
        rc, out = run_capture(["nvidia-smi"], timeout=30)
        lines += ["## nvidia-smi", out.strip(), ""]
        rc2, name = run_capture(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(args.gpu_id)],
            timeout=15)
        if rc2 == 0 and name.strip():
            info["gpu_name"] = name.strip().splitlines()[0]
    else:
        lines += ["## nvidia-smi", "(nvidia-smi not found on this host)", ""]

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / "run_env.txt").write_text("\n".join(lines) + "\n")
    return info


# --------------------------------------------------------------------------------------
# Dry run: print the exact commands without launching anything
# --------------------------------------------------------------------------------------
def dry_run(modes, args):
    out = [f"# DRY RUN - {now_utc()}", ""]
    out.append(f"Greedy enforcement: {greedy_enforcement_note()}")
    out.append(f"vllm on PATH: {shutil.which('vllm') is not None}   "
               f"nvidia-smi: {HAVE_NVIDIA_SMI}")
    out.append("")
    for m in modes:
        out.append(f"## {m['label']}  ({m['name']})")
        out.append("serve:")
        out.append("  " + shlex.join(map(str, build_serve_cmd(
            m, args, args.gpu_mem_util, args.max_model_len))))
        out.append("bench:")
        bj = str(Path(args.results_dir) / m["name"] / "bench.json")
        out.append("  " + shlex.join(map(str, build_bench_cmd(m, args, bj))))
        out.append("")
    text = "\n".join(out)
    print(text)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / "dry_run_preview.txt").write_text(text + "\n")
    return text


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Three-mode speculative-decoding benchmark for Qwen3.5-9B (vLLM).")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--draft-model", default="meta-llama/Llama-3.2-1B-Instruct",
                   help="Draft model for mode C (TLI).")
    p.add_argument("--dataset-path", default="./ShareGPT_V3_unfiltered_cleaned_split.json")
    p.add_argument("--results-dir", default="results")
    # Fixed-parameter defaults straight from the task spec:
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--concurrency", type=int, default=1, help="--max-concurrency (batch-1 regime).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--num-spec-tokens", type=int, default=3,
                   help="num_speculative_tokens for BOTH spec modes (kept equal for fairness).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--modes", default="A,B,C",
                   help="Comma list subset of A,B,C (or full mode names).")
    p.add_argument("--startup-timeout", type=int, default=1800,
                   help="Seconds to wait for /health (nightly model load can be slow).")
    p.add_argument("--bench-timeout", type=int, default=7200)
    p.add_argument("--gpu-free-timeout", type=int, default=180)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the serve/bench commands and exit (launch nothing).")
    return p.parse_args(argv)


def select_modes(all_modes, spec: str):
    alias = {"A": "A_baseline", "B": "B_mtp", "C": "C_tli"}
    wanted = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        wanted.append(alias.get(tok.upper(), tok))
    return [m for m in all_modes if m["name"] in wanted] or all_modes


def main(argv=None):
    args = parse_args(argv)
    modes = build_modes(args.num_spec_tokens, args.draft_model)
    modes = select_modes(modes, args.modes)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run(modes, args)
        return 0

    if shutil.which("vllm") is None:
        log("ERROR: `vllm` not found on PATH. Install vLLM nightly first (see README). "
            "Nothing was run. Use --dry-run to preview the commands.")
        return 2
    if not Path(args.dataset_path).exists():
        log(f"ERROR: dataset not found at {args.dataset_path}. Download it (see README):\n"
            "  wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
            "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json")
        return 2

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    log("recording run environment (nvidia-smi, vllm --version) ...")
    env_info = record_env(args)
    log(f"vLLM {env_info['vllm_version']} | GPU {env_info['gpu_name']}")

    results = []
    for mode in modes:
        log(f"===== MODE {mode['name']} :: {mode['label']} =====")
        try:
            results.append(run_mode(mode, args, env))
        except KeyboardInterrupt:
            log("interrupted by user; writing partial summary.")
            break
        except Exception as e:  # pragma: no cover
            log(f"[{mode['name']}] harness error: {e}")
            results.append({"mode": mode["name"], "label": mode["label"],
                            "spec": mode["spec"], "status": "FAILED",
                            "reason": f"harness error: {e}", "output_throughput": None,
                            "tpot_ms": None, "ttft_ms": None, "tau": None,
                            "tau_source": None, "notes": [],
                            "speculative_config": mode["speculative_config"]})

    summary_md = build_summary(results, args, env_info)
    (Path(args.results_dir) / "summary.md").write_text(summary_md)
    (Path(args.results_dir) / "summary.json").write_text(
        json.dumps({"generated": now_utc(), "env": env_info,
                    "args": vars(args), "results": results}, indent=2))
    print("\n" + "=" * 80)
    print(summary_md)
    log(f"wrote {args.results_dir}/summary.md and summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
