#!/usr/bin/env python3
"""SLEM arm (HF Transformers Universal Assisted Decoding) for the speculative-decoding benchmark.

Engine: HuggingFace transformers (pinned 5.14.1), NOT vLLM. SLEM (string-level detok ->
re-tok cross-vocab drafting) does not exist in vLLM; `use_heterogeneous_vocab` there is
Token-Level Intersection only. HF selects the SLEM path implicitly: passing
`assistant_model` + `tokenizer` + `assistant_tokenizer` with `do_sample=False` routes to
`AssistedCandidateGeneratorDifferentTokenizers` (utils.py:1040-1071 in 5.14.1).

Arms (same engine, only compare against each other -- NEVER against the vLLM numbers):
  A_hf  target-only greedy baseline
  D_hf  SLEM via UAD

Version-locked facts (verified against transformers 5.14.1 source; the version check below
exists because these are load-bearing):
  - `num_assistant_tokens`, `num_assistant_tokens_schedule` and
    `assistant_confidence_threshold` are read from the DRAFT model's generation_config
    (candidate_generator.py:122-126), NOT from the main generate() kwargs. We set them on
    the draft config; the generate() kwargs are passed too but are documentation only.
  - Setting `assistant_confidence_threshold=None` does NOT disable the confidence early
    stop: `update(defaults_only=True)` re-fills None with the 0.4 default
    (configuration_utils.py:1326). The gate is `is not None and > 0` (utils.py:1349-1353),
    so 0.0 is the correct "off" value. Left at 0.4 the draft silently stops below k.
  - Retokenization is already windowed upstream (target_lookbehind=10): O(k) per
    iteration, O(context) only on the first iteration of each prompt.
  - The assisted loop calls update_candidate_strategy() every iteration (utils.py:3819),
    which is what closes each per-iteration profile row.

Instrumentation (no site-packages edits):
  - InstrumentedUAD subclasses AssistedCandidateGeneratorDifferentTokenizers and is
    swapped in by rebinding the class name inside transformers.generation.utils, so
    _get_candidate_generator instantiates ours.
  - TimedTokenizer proxies wrap decode (detok_ms) and __call__ (retok_ms +
    retok_input_chars/tokens); they are what gets passed as tokenizer/assistant_tokenizer.
  - _get_tokens_diag (the O(window^2) pure-Python LCS alignment, called up to twice per
    iteration) is timed separately as align_ms.
  - The draft model's .generate is wrapped for draft_ms and n_drafted.
  - target_verify_ms = get_candidates return -> update_candidate_strategy entry, which is
    the target forward + acceptance matching. Device-synchronized at every boundary.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REQUIRED_TRANSFORMERS = "5.14.1"

SMOKE_PROMPTS = [
    "The history of the Roman Empire begins with the founding of the city of Rome,",
    "In Python, the difference between a list and a tuple is",
    "Photosynthesis is the process by which plants",
    "The lighthouse keeper climbed the spiral staircase for the last time,",
    "To make a simple tomato pasta sauce, start by",
]


def fail(msg: str, code: int = 2):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def check_transformers_version():
    import transformers

    if transformers.__version__ != REQUIRED_TRANSFORMERS:
        fail(
            f"transformers=={transformers.__version__} but this harness is source-locked to "
            f"{REQUIRED_TRANSFORMERS} (instrumentation subclasses internals). "
            f"Install the pin: pip install transformers=={REQUIRED_TRANSFORMERS}"
        )


def check_repo_access(repo_id: str):
    """Fail fast if HF auth does not resolve the repo (Llama-3.2 is gated)."""
    from huggingface_hub import auth_check
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        auth_check(repo_id)
    except GatedRepoError:
        fail(
            f"'{repo_id}' is a gated repo and the current HF credentials do not grant access. "
            f"Run `hf auth login` (or set HF_TOKEN) with an account that has accepted the "
            f"license at https://huggingface.co/{repo_id}, then retry."
        )
    except RepositoryNotFoundError:
        fail(
            f"'{repo_id}' not found, or it is gated/private and no HF credentials are set at all. "
            f"Run `hf auth login` (or set HF_TOKEN) and make sure the license is accepted."
        )
    except Exception as e:  # network down etc. -- only warn, the weights may be cached
        print(f"WARNING: could not verify access to {repo_id} ({type(e).__name__}: {e}); "
              f"continuing, will fail at download time if truly inaccessible.")


# --------------------------------------------------------------------------------------
# Profiler + instrumented classes
# --------------------------------------------------------------------------------------
class SpecProfiler:
    """Accumulates one row per speculation iteration for the current prompt."""

    def __init__(self, sync):
        self.sync = sync
        self.rows: list[dict] = []
        self.cur: dict | None = None
        self.generator_classes: list[type] = []
        self._t_iter0 = 0.0
        self._t_verify0 = 0.0
        self.warnings: list[str] = []

    def on_generator_created(self, gen):
        self.generator_classes.append(type(gen))

    def begin_iteration(self, context_len: int, k: int):
        if self.cur is not None:
            self.warnings.append(f"iteration at context_len={self.cur['context_len']} never closed")
            self.rows.append(self.cur)
        self.sync()
        self._t_iter0 = time.perf_counter()
        self.cur = dict(
            iter_idx=len(self.rows), k=k, context_len=context_len,
            iter_total_ms=0.0, draft_ms=0.0, detok_ms=0.0, retok_ms=0.0, align_ms=0.0,
            target_verify_ms=0.0, candidates_ms=0.0, n_drafted=0, n_accepted=None,
            n_target_candidate_tokens=0, retok_input_chars=0, retok_input_tokens=0,
            n_detok_calls=0, n_retok_calls=0, n_align_calls=0,
        )

    def mark_candidates_done(self, n_target_candidate_tokens: int, t0: float):
        # called from get_candidates AFTER a sync
        now = time.perf_counter()
        self.cur["candidates_ms"] = (now - t0) * 1e3
        self.cur["n_target_candidate_tokens"] = n_target_candidate_tokens
        self._t_verify0 = now

    def end_iteration(self, n_accepted: int):
        self.sync()
        now = time.perf_counter()
        if self.cur is None:
            self.warnings.append("update_candidate_strategy without open iteration")
            return
        self.cur["target_verify_ms"] = (now - self._t_verify0) * 1e3
        self.cur["iter_total_ms"] = (now - self._t_iter0) * 1e3
        self.cur["n_accepted"] = n_accepted
        self.rows.append(self.cur)
        self.cur = None

    def add(self, field, ms, **counts):
        if self.cur is None:
            return
        self.cur[field] += ms
        for key, val in counts.items():
            self.cur[key] += val

    def finish_prompt(self):
        if self.cur is not None:
            self.warnings.append("prompt ended with an unclosed iteration (discarded)")
            self.cur = None
        rows, self.rows = self.rows, []
        return rows


class TimedTokenizer:
    """Delegating proxy: times decode() as detok and __call__() as retok."""

    def __init__(self, tok, prof: SpecProfiler):
        self._tok = tok
        self._prof = prof

    def decode(self, *args, **kwargs):
        self._prof.sync()  # decode() pulls GPU ids to CPU; don't blame queued kernels
        t0 = time.perf_counter()
        out = self._tok.decode(*args, **kwargs)
        self._prof.add("detok_ms", (time.perf_counter() - t0) * 1e3, n_detok_calls=1)
        return out

    def __call__(self, text, *args, **kwargs):
        t0 = time.perf_counter()
        enc = self._tok(text, *args, **kwargs)
        ms = (time.perf_counter() - t0) * 1e3
        ids = enc["input_ids"]
        n_tok = int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids)
        n_chars = len(text) if isinstance(text, str) else sum(len(t) for t in text)
        self._prof.add("retok_ms", ms, n_retok_calls=1,
                       retok_input_chars=n_chars, retok_input_tokens=n_tok)
        return enc

    def __getattr__(self, name):
        return getattr(self._tok, name)


def build_instrumented_class():
    """Defined in a function so transformers is only imported after the env checks."""
    import torch  # noqa: F401
    from transformers.generation.candidate_generator import (
        AssistedCandidateGeneratorDifferentTokenizers as _UAD,
    )

    class InstrumentedUAD(_UAD):
        profiler: SpecProfiler = None  # set before each D_hf generate()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            type(self).profiler.on_generator_created(self)

        def get_candidates(self, input_ids, **kwargs):
            prof = type(self).profiler
            prof.begin_iteration(context_len=int(input_ids.shape[-1]),
                                 k=int(self.num_assistant_tokens))
            t0 = time.perf_counter()
            result = super().get_candidates(input_ids, **kwargs)
            prof.sync()
            n_cand = int(result[0].shape[-1]) - int(input_ids.shape[-1])
            prof.mark_candidates_done(n_target_candidate_tokens=n_cand, t0=t0)
            return result

        def update_candidate_strategy(self, input_ids, scores, num_matches):
            type(self).profiler.end_iteration(n_accepted=int(num_matches))
            return super().update_candidate_strategy(input_ids, scores, num_matches)

        @staticmethod
        def _get_tokens_diag(prompt, prompt_plus_new_tokens):
            prof = InstrumentedUAD.profiler
            prof.sync()  # the LCS loop calls .item() on device tensors
            t0 = time.perf_counter()
            out = _UAD._get_tokens_diag(prompt, prompt_plus_new_tokens)
            prof.add("align_ms", (time.perf_counter() - t0) * 1e3, n_align_calls=1)
            return out

    return InstrumentedUAD


def wrap_draft_generate(draft, prof: SpecProfiler):
    orig_generate = draft.generate

    def timed_generate(*args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        n_in = int(ids.shape[-1])
        prof.sync()
        t0 = time.perf_counter()
        out = orig_generate(*args, **kwargs)
        prof.sync()
        seq = getattr(out, "sequences", out)
        prof.add("draft_ms", (time.perf_counter() - t0) * 1e3,
                 n_drafted=int(seq.shape[-1]) - n_in)
        return out

    draft.generate = timed_generate


# --------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------
def scrub_sampling(model):
    """Force pure-greedy configs; Qwen ships do_sample=True + temperature in its defaults."""
    gc = model.generation_config
    gc.do_sample = False
    for attr in ("temperature", "top_p", "top_k", "min_p"):
        setattr(gc, attr, None)


def bucket_of(n_prompt_tokens: int) -> str:
    if n_prompt_tokens < 512:
        return "short"
    if n_prompt_tokens >= 4096:
        return "long"
    return "mid"


def load_prompts(args, target_tok):
    if args.dataset_path and Path(args.dataset_path).exists():
        import random

        data = json.loads(Path(args.dataset_path).read_text())
        texts = []
        for conv in data:
            turns = conv.get("conversations", [])
            if turns and turns[0].get("from") == "human" and turns[0].get("value", "").strip():
                texts.append(turns[0]["value"])
        random.Random(args.seed).shuffle(texts)
        prompts = texts[: args.num_prompts]
        src = f"ShareGPT ({args.dataset_path}), seed {args.seed}"
    else:
        prompts = (SMOKE_PROMPTS * ((args.num_prompts // len(SMOKE_PROMPTS)) + 1))[: args.num_prompts]
        src = "built-in smoke prompts"
        if args.dataset_path:
            print(f"WARNING: dataset {args.dataset_path} not found; using {src}")
    return prompts, src


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--k", type=int, default=4, help="num_assistant_tokens")
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-path", default=None, help="ShareGPT json; built-in prompts if absent")
    ap.add_argument("--results-dir", default="results/slem_smoke")
    ap.add_argument("--arms", default="A_hf,D_hf")
    args = ap.parse_args()

    check_transformers_version()
    for repo in (args.target, args.draft):
        check_repo_access(repo)
    print(f"auth ok: {args.target}, {args.draft}")

    import torch
    import transformers
    import transformers.generation.utils as hf_gen_utils
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        fail("--device cuda requested but torch.cuda.is_available() is False")
    if device.type == "cuda":
        sync = lambda: torch.cuda.synchronize(device)  # noqa: E731
    elif device.type == "mps":
        sync = torch.mps.synchronize
    else:
        sync = lambda: None  # noqa: E731
    dtype = getattr(torch, args.dtype)

    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"loading target {args.target} ({args.dtype}, {device}) ...")
    target_tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    scrub_sampling(target)

    print(f"loading draft {args.draft} ...")
    draft_tok = AutoTokenizer.from_pretrained(args.draft)
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=dtype).to(device).eval()
    scrub_sampling(draft)

    # The operative location for these knobs (candidate_generator.py:122-126 reads the
    # DRAFT config; None would be re-defaulted, hence 0.0 to disable the confidence stop).
    draft.generation_config.num_assistant_tokens = args.k
    draft.generation_config.num_assistant_tokens_schedule = "constant"
    draft.generation_config.assistant_confidence_threshold = 0.0

    prompts, prompt_src = load_prompts(args, target_tok)
    print(f"{len(prompts)} prompts from {prompt_src}")

    prof = SpecProfiler(sync)
    InstrumentedUAD = build_instrumented_class()
    InstrumentedUAD.profiler = prof
    hf_gen_utils.AssistedCandidateGeneratorDifferentTokenizers = InstrumentedUAD
    wrap_draft_generate(draft, prof)

    arms = [a.strip() for a in args.arms.split(",")]
    results = {a: [] for a in arms}
    jsonl_path = outdir / "slem_profile.jsonl"
    jsonl = jsonl_path.open("w")

    for arm in arms:
        wall0 = time.perf_counter()
        gen_tokens = 0
        for pi, prompt in enumerate(prompts):
            enc = target_tok(prompt, return_tensors="pt").to(device)
            n_prompt = int(enc["input_ids"].shape[-1])
            common = dict(do_sample=False, max_new_tokens=args.max_new_tokens,
                          pad_token_id=target_tok.eos_token_id)
            sync()
            t0 = time.perf_counter()
            if arm == "A_hf":
                out = target.generate(**enc, **common)
            elif arm == "D_hf":
                out = target.generate(
                    **enc,
                    assistant_model=draft,
                    tokenizer=TimedTokenizer(target_tok, prof),
                    assistant_tokenizer=TimedTokenizer(draft_tok, prof),
                    # documentation mirrors; the draft-config copies above are operative
                    num_assistant_tokens=args.k,
                    num_assistant_tokens_schedule="constant",
                    assistant_confidence_threshold=0.0,
                    **common,
                )
            else:
                fail(f"unknown arm {arm}")
            sync()
            wall_ms = (time.perf_counter() - t0) * 1e3
            seq = out[0] if isinstance(out, torch.Tensor) else out.sequences[0]
            new_ids = seq[n_prompt:].tolist()
            gen_tokens += len(new_ids)
            rec = dict(arm=arm, prompt_idx=pi, prompt_tokens=n_prompt,
                       prompt_bucket=bucket_of(n_prompt), wall_ms=wall_ms,
                       new_tokens=len(new_ids), output_ids=new_ids)
            if arm == "D_hf":
                rows = prof.finish_prompt()
                rec["iters"] = rows
                for row in rows:
                    jsonl.write(json.dumps(dict(arm=arm, prompt_idx=pi,
                                                prompt_bucket=rec["prompt_bucket"], **row)) + "\n")
            results[arm].append(rec)
            print(f"  {arm} prompt {pi}: {len(new_ids)} tok in {wall_ms:.0f} ms")
        wall = time.perf_counter() - wall0
        print(f"{arm}: {gen_tokens} tokens in {wall:.1f}s = {gen_tokens / wall:.2f} tok/s")
    jsonl.close()

    # ----------------------------------------------------------------------------------
    # Smoke assertions
    # ----------------------------------------------------------------------------------
    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    if "D_hf" in results:
        d_recs = results["D_hf"]
        n_gen = len(prof.generator_classes)
        klass = prof.generator_classes[0] if prof.generator_classes else None
        from transformers.generation.candidate_generator import (
            AssistedCandidateGeneratorDifferentTokenizers as UAD_base,
            UniversalSpeculativeDecodingGenerator as USD,
        )
        ok = (n_gen == len(d_recs) and klass is not None
              and issubclass(klass, UAD_base) and not issubclass(klass, USD))
        check("1 candidate generator class", ok,
              f"{n_gen} generators; class={klass.__name__ if klass else None} "
              f"(base={klass.__mro__[1].__name__ if klass else None}; not USD/TLI path)")

        all_rows = [row for rec in d_recs for row in rec["iters"]]
        bad_k = [(r["iter_idx"], r["n_drafted"]) for r in all_rows if r["n_drafted"] != args.k]
        check("2 n_drafted == k every iteration",
              not bad_k and all_rows,
              f"{len(all_rows)} iterations, k={args.k}"
              + (f", violations={bad_k[:10]}" if bad_k else ", no violations"))

        zeros = {f: sum(1 for r in all_rows if r[f] <= 0.0)
                 for f in ("detok_ms", "retok_ms", "align_ms")}
        check("3 detok/retok/align separately populated",
              all_rows and all(v == 0 for v in zeros.values()),
              f"iterations with zero value: {zeros}")

        if "A_hf" in results:
            diffs = []
            for a, d in zip(results["A_hf"], d_recs):
                if a["output_ids"] != d["output_ids"]:
                    div = next((i for i, (x, y) in enumerate(zip(a["output_ids"], d["output_ids"]))
                                if x != y), min(len(a["output_ids"]), len(d["output_ids"])))
                    diffs.append((a["prompt_idx"], div, len(a["output_ids"]), len(d["output_ids"])))
            check("4 losslessness (D_hf == A_hf token-identical)", not diffs,
                  "all prompts identical" if not diffs else
                  f"diverged (prompt, first_div_idx, lenA, lenD): {diffs}")

        print("\nretok_input_tokens per iteration (assertion 5):")
        for rec in d_recs:
            toks = [r["retok_input_tokens"] for r in rec["iters"]]
            print(f"  prompt {rec['prompt_idx']} ({rec['prompt_tokens']} prompt tok): {toks}")

        if prof.warnings:
            print(f"profiler warnings: {prof.warnings}")

    # ----------------------------------------------------------------------------------
    # Summary artifacts
    # ----------------------------------------------------------------------------------
    def med(rows, f):
        vals = [r[f] for r in rows]
        return statistics.median(vals) if vals else None

    summary = dict(
        engine=f"transformers {transformers.__version__}",
        device=str(device), dtype=args.dtype,
        target=args.target, draft=args.draft, k=args.k,
        prompts=dict(n=len(prompts), source=prompt_src,
                     max_new_tokens=args.max_new_tokens, seed=args.seed),
        knob_placement="draft.generation_config (candidate_generator.py:122-126); "
                       "assistant_confidence_threshold=0.0 (None re-defaults to 0.4)",
        arms={},
        checks=[dict(name=n, passed=p, detail=d) for n, p, d in checks],
        profiler_warnings=prof.warnings,
    )
    for arm, recs in results.items():
        tokens = sum(r["new_tokens"] for r in recs)
        wall_s = sum(r["wall_ms"] for r in recs) / 1e3
        entry = dict(gen_tokens=tokens, wall_s=round(wall_s, 2),
                     tok_per_s=round(tokens / wall_s, 2) if wall_s else None)
        if arm == "D_hf":
            rows = [row for r in recs for row in r["iters"]]
            if rows:
                entry.update(
                    iterations=len(rows),
                    tau_mean=round(1 + statistics.mean(r["n_accepted"] for r in rows), 3),
                    median_ms=dict(iter_total=med(rows, "iter_total_ms"), draft=med(rows, "draft_ms"),
                                   detok=med(rows, "detok_ms"), retok=med(rows, "retok_ms"),
                                   align=med(rows, "align_ms"),
                                   target_verify=med(rows, "target_verify_ms")),
                )
        summary["arms"][arm] = entry
    (outdir / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {jsonl_path} and {outdir / 'smoke_summary.json'}")
    if any(not p for _, p, _ in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
