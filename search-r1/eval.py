"""Evaluate a Search-R1 model against an SGLang server.

Mirrors retool/eval.py: assumes you have already started an SGLang server
(see search-r1/sglang_serve.sh) AND a local retrieval server (see
search-r1/local_search_server.py + Search-R1's retrieval_launch.sh).

Run with:
    bash search-r1/eval.sh
or directly:
    python search-r1/eval.py --model-path ... --dataset .../test.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import socket
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Search-R1 model against an already-running SGLang server "
            "(start it with search-r1/sglang_serve.sh) and an already-running local "
            "retrieval server (Search-R1's retrieval_launch.sh)."
        )
    )
    parser.add_argument("--model-path", required=True, help="HF model path (used for the chat template + summary)")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path. Defaults to --model-path.")
    parser.add_argument("--dataset", required=True, help="Search-R1 test parquet (or jsonl/HF dataset name)")
    parser.add_argument("--split", default="test", help="HF dataset split (only used when --dataset is a HF name)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples")
    parser.add_argument("--data-source-filter", default=None,
                        help="Comma-separated subset of data_source values to keep (e.g. 'nq,hotpotqa').")
    parser.add_argument("-n", "--num-samples", type=int, default=1, help="Traces per prompt (>=1)")

    parser.add_argument("--host", default="127.0.0.1", help="SGLang server host")
    parser.add_argument("--port", type=int, default=30000, help="SGLang server port")
    parser.add_argument("--server-wait-timeout", type=int, default=60,
                        help="Seconds to wait for the SGLang port to accept connections")
    parser.add_argument("--request-timeout", type=int, default=1800, help="HTTP timeout per /generate call")

    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens per turn")
    parser.add_argument("--max-context-len", type=int, default=8192,
                        help="Hard cap on input_tokens + max_new_tokens per /generate call")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--max-turns", type=int, default=4, help="Max search rounds per trace")
    parser.add_argument("--topk", type=int, default=3, help="topk for the retrieval server")
    parser.add_argument("--search-url", default="http://127.0.0.1:8000/retrieve",
                        help="URL of the local retrieval server")
    parser.add_argument("--search-backend", choices=["local", "google"], default="local")
    parser.add_argument("--google-api-key", default=None, help="serper.dev API key (when --search-backend=google)")
    parser.add_argument("--google-snippet-only", action="store_true", default=True)

    parser.add_argument("--max-concurrent", type=int, default=8,
                        help="Max concurrent in-flight samples")
    parser.add_argument("--sample-timeout", type=int, default=600,
                        help="Hard wall-clock budget per trace (all turns combined)")

    parser.add_argument("--output", default=None, help="JSONL output for per-example results")
    parser.add_argument("--summary-output", default=None, help="JSON output for the final summary")
    parser.add_argument("--print-turns", action="store_true", help="Print per-turn assistant/observation text")
    parser.add_argument("--debug-trace", action="store_true",
                        help="Print per-turn START/GEN/SEARCH timings")

    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("-n/--num-samples must be at least 1")
    return args


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_tokenizer(tokenizer_path: str):
    transformers = importlib.import_module("transformers")
    return transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _load_runtime():
    """Import the eval-side helpers (search/postprocess/execute_predictions)."""
    return importlib.import_module("eval_generate_with_search")


def _load_grader():
    """Return qa_em_format.compute_score_em."""
    module = importlib.import_module("qa_em_format")
    return module.compute_score_em


def _load_dataset_records(args: argparse.Namespace):
    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        if dataset_path.suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(dataset_path), split="train")
        elif dataset_path.suffix == ".jsonl":
            dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        else:
            dataset = load_dataset(str(dataset_path), split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    if args.data_source_filter:
        wanted = {s.strip() for s in args.data_source_filter.split(",") if s.strip()}
        dataset = dataset.filter(lambda r: r.get("data_source") in wanted)

    if args.limit is not None:
        limit = min(args.limit, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


# ---------------------------------------------------------------------------
# Prompt / extraction helpers
# ---------------------------------------------------------------------------


def _row_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Extract the chat messages for a Search-R1 verl-style row."""
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return [
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
            for m in prompt
            if isinstance(m, dict)
        ]
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    raise ValueError(f"Unsupported prompt format in row: {type(prompt)}")


def _row_ground_truth(row: dict[str, Any]) -> dict[str, list[str]]:
    """Extract the ground_truth dict {target: [...]} from a verl-style row."""
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if isinstance(gt, dict) and "target" in gt:
            target = gt["target"]
            if isinstance(target, str):
                target = [target]
            return {"target": list(target)}
    # Fallbacks for ad-hoc datasets.
    for key in ("answer", "answers", "golden_answers", "gt"):
        if key in row:
            value = row[key]
            if isinstance(value, str):
                return {"target": [value]}
            if isinstance(value, list):
                return {"target": [str(v) for v in value]}
    raise ValueError("Could not extract ground_truth from row")


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply the model's chat template (returning text)."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Async HTTP / server helpers (lightweight clone of retool/eval.py)
# ---------------------------------------------------------------------------


async def _post_json_async(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    body = json.dumps(payload).encode("utf-8")
    req = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode() + body

    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=10)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"HTTP connect to {host}:{port} timed out after 10s") from exc
    try:
        writer.write(req)
        await asyncio.wait_for(writer.drain(), timeout=10)
        response_bytes = await asyncio.wait_for(reader.read(-1), timeout=timeout)
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=5)
        except Exception:
            pass

    sep = response_bytes.find(b"\r\n\r\n")
    if sep == -1:
        raise ValueError(f"Malformed HTTP response (no header separator): {response_bytes[:200]}")
    return json.loads(response_bytes[sep + 4 :])


async def _post_generate(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/generate"
    return await _post_json_async(url, payload, args.request_timeout)


def _wait_for_port(host: str, port: int, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(
        f"Timed out waiting for SGLang server on {host}:{port} after {timeout}s. "
        f"Is the server running? Last socket error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Multi-turn rollout
# ---------------------------------------------------------------------------


async def _generate_one_with_search(
    args: argparse.Namespace,
    tokenizer: Any,
    base_prompt: str,
    runtime: Any,
    debug_tag: str = "",
) -> dict[str, Any]:
    trace_start = time.time()
    response_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    search_count = 0
    timed_out = False
    finish_reason: str | None = None

    progress: dict[str, Any] = {"current_turn": -1, "current_stage": "init",
                                 "stage_started_at": trace_start}

    try:
        for turn_index in range(args.max_turns):
            current_text = base_prompt + "".join(response_parts)
            input_ids = tokenizer(current_text, add_special_tokens=False)["input_ids"]
            context_budget = max(0, args.max_context_len - len(input_ids) - 16)
            turn_max_new = min(args.max_new_tokens, context_budget)
            if turn_max_new <= 0:
                if args.debug_trace:
                    print(
                        f"{debug_tag} turn={turn_index} CONTEXT-FULL input_tokens={len(input_ids)} "
                        f"max_context_len={args.max_context_len} — stopping",
                        flush=True,
                    )
                break

            payload = {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": turn_max_new,
                    # Stop right after the model emits </search>, so we can run
                    # the search and continue from the same place.
                    "stop": ["</search>", "</answer>"],
                },
            }
            if args.debug_trace:
                print(
                    f"{debug_tag} turn={turn_index} GEN-> input_tokens={len(input_ids)} "
                    f"elapsed={time.time() - trace_start:.1f}s",
                    flush=True,
                )
            gen_t0 = time.time()
            progress["current_turn"] = turn_index
            progress["current_stage"] = "gen"
            progress["stage_started_at"] = gen_t0
            output = await _post_generate(args, payload)
            cur_response = runtime.postprocess_responses(output["text"])
            finish_reason = output.get("meta_info", {}).get("finish_reason", {}).get("type")
            if args.debug_trace:
                print(
                    f"{debug_tag} turn={turn_index} GEN<- gen_s={time.time() - gen_t0:.1f} "
                    f"resp_chars={len(cur_response)} finish={finish_reason}",
                    flush=True,
                )

            response_parts.append(cur_response)
            turn_record = {"turn_index": turn_index, "assistant": cur_response}

            if finish_reason == "abort":
                turn_record["aborted"] = True
                turns.append(turn_record)
                break

            if finish_reason == "length":
                turns.append(turn_record)
                break

            search_t0 = time.time()
            progress["current_stage"] = "search"
            progress["stage_started_at"] = search_t0
            next_obs, done = await runtime.execute_predictions(cur_response)
            if args.debug_trace:
                action, _ = runtime.postprocess_predictions(cur_response)
                print(
                    f"{debug_tag} turn={turn_index} SEARCH<- search_s={time.time() - search_t0:.1f} "
                    f"action={action} done={done} obs_chars={len(next_obs)}",
                    flush=True,
                )
            turn_record["done"] = done
            if done:
                turns.append(turn_record)
                break


            # Increment search_count only when <tool_response> is present in the observation (successful search)
            if "<tool_response>" in next_obs:
                search_count += 1

            response_parts.append(next_obs)
            turn_record["observation"] = next_obs
            turns.append(turn_record)
    except asyncio.CancelledError:
        timed_out = True
        if debug_tag:
            stage = progress.get("current_stage", "init")
            stage_dt = time.time() - progress.get("stage_started_at", trace_start)
            print(
                f"{debug_tag} CANCELLED stage={stage} turn={progress.get('current_turn', -1)} "
                f"stage_s={stage_dt:.1f} elapsed={time.time() - trace_start:.1f}s "
                f"completed_turns={len(turns)} searches={search_count}",
                flush=True,
            )

    return {
        "response": "".join(response_parts),
        "turns": turns,
        "search_count": search_count,
        "timed_out": timed_out,
        "finish_reason": finish_reason,
        "timeout_stage": progress.get("current_stage") if timed_out else None,
        "timeout_turn": progress.get("current_turn") if timed_out else None,
    }


# ---------------------------------------------------------------------------
# Scoring + per-trace helpers
# ---------------------------------------------------------------------------


def _score_response(grader: Any, prompt: str, ground_truth: dict[str, Any], response: str,
                    format_score: float) -> dict[str, Any]:
    score = grader(
        solution_str=prompt + response,
        ground_truth=ground_truth,
        format_score=format_score,
    )
    score = float(score)
    return {
        "score": score,
        "acc": score >= 1.0,
        "response": response,
    }


def _print_trace_turns(example_index: int, trace: dict[str, Any]) -> None:
    print(f"\n=== example {example_index} trace {trace['trace_index']} ===")
    print(
        f"searches={trace['search_count']} score={trace['score']:.3f} "
        f"acc={int(trace['acc'])} timed_out={trace['timed_out']}"
    )
    for turn in trace.get("turns", []):
        print(f"\n--- turn {turn['turn_index']} assistant ---")
        print(turn.get("assistant", ""))
        if "observation" in turn:
            print("\n--- observation ---")
            print(turn["observation"])


# ---------------------------------------------------------------------------
# Main eval driver
# ---------------------------------------------------------------------------


def _apply_runtime_overrides(args: argparse.Namespace, runtime: Any) -> None:
    """Push CLI overrides into the runtime's SEARCH_R1_CONFIGS."""
    cfg = runtime.SEARCH_R1_CONFIGS
    cfg["max_turns"] = args.max_turns
    cfg["topk"] = args.topk
    cfg["search_backend"] = args.search_backend
    cfg["local"]["search_url"] = args.search_url
    if args.google_api_key:
        cfg["google"]["api_key"] = args.google_api_key
    cfg["google"]["snippet_only"] = args.google_snippet_only


def main() -> None:
    args = parse_args()
    tokenizer = _load_tokenizer(args.tokenizer_path or args.model_path)
    runtime = _load_runtime()
    grader = _load_grader()
    _apply_runtime_overrides(args, runtime)

    dataset = _load_dataset_records(args)
    output_path = Path(args.output) if args.output else None

    print(
        f"Connecting to SGLang server at http://{args.host}:{args.port} "
        f"(waiting up to {args.server_wait_timeout}s).",
        flush=True,
    )
    _wait_for_port(args.host, args.port, timeout=args.server_wait_timeout)
    print(f"SGLang server is reachable at http://{args.host}:{args.port}", flush=True)

    async def _run_eval() -> dict[str, Any]:
        num_examples = len(dataset)
        semaphore = asyncio.Semaphore(args.max_concurrent)
        output_file = output_path.open("w", encoding="utf-8") if output_path else None
        output_lock = asyncio.Lock()

        rows = [(i, dataset[i]) for i in range(num_examples)]

        done_examples = 0
        num_correct = 0.0
        total_score = 0.0
        total_timeouts = 0
        examples_with_any_timeout = 0

        async def _safe_generate(ex_index: int, sample_idx: int, prompt_text: str) -> dict[str, Any]:
            tag = f"[ex{ex_index} s{sample_idx}]"
            async with semaphore:
                t0 = time.time()
                try:
                    return await asyncio.wait_for(
                        _generate_one_with_search(args, tokenizer, prompt_text, runtime, debug_tag=tag),
                        timeout=args.sample_timeout,
                    )
                except asyncio.TimeoutError:
                    print(
                        f"{tag} SAMPLE-TIMEOUT (hard) after {args.sample_timeout}s "
                        f"(wall={time.time() - t0:.1f}s)",
                        flush=True,
                    )
                    return {
                        "response": "",
                        "turns": [],
                        "search_count": 0,
                        "timed_out": True,
                        "finish_reason": None,
                        "timeout_stage": "unknown",
                        "timeout_turn": -1,
                    }
                except Exception as exc:
                    print(
                        f"{tag} ERROR {type(exc).__name__}: {exc} (wall={time.time() - t0:.1f}s)",
                        flush=True,
                    )
                    return {
                        "response": "",
                        "turns": [],
                        "search_count": 0,
                        "timed_out": True,
                        "finish_reason": None,
                        "timeout_stage": "error",
                        "timeout_turn": -1,
                    }

        async def _process_example(ex_index: int, row: dict[str, Any]) -> None:
            nonlocal done_examples, num_correct, total_score
            nonlocal total_timeouts, examples_with_any_timeout


            messages = _row_messages(row)
            ground_truth = _row_ground_truth(row)
            # Use the new tool-based format for prompt rendering
            prompt_text = runtime.format_conversation_with_tools(
                prompt=messages,
                system_prompt=None,
                messages=None
            )

            generations = await asyncio.gather(*[
                _safe_generate(ex_index, i, prompt_text) for i in range(args.num_samples)
            ])

            traces: list[dict[str, Any]] = []
            for sample_idx, generation in enumerate(generations):
                trace = _score_response(
                    grader, prompt_text, ground_truth, generation["response"],
                    runtime.SEARCH_R1_CONFIGS["format_score"],
                )
                trace["trace_index"] = sample_idx
                trace["turns"] = generation["turns"]
                trace["search_count"] = generation["search_count"]
                trace["timed_out"] = generation["timed_out"]
                trace["finish_reason"] = generation.get("finish_reason")
                if trace["timed_out"]:
                    trace["timeout_stage"] = generation.get("timeout_stage")
                    trace["timeout_turn"] = generation.get("timeout_turn")
                traces.append(trace)
                if args.print_turns:
                    _print_trace_turns(ex_index, trace)

            valid_traces = [t for t in traces if not t["timed_out"]]
            num_timeouts = len(traces) - len(valid_traces)
            all_timed_out = len(valid_traces) == 0

            avg_score = sum(t["score"] for t in traces) / len(traces)
            avg_acc = sum(int(t["acc"]) for t in traces) / len(traces)
            if all_timed_out:
                avg_acc_excl_timeout = 0.0
                avg_score_excl_timeout = 0.0
            else:
                avg_acc_excl_timeout = sum(int(t["acc"]) for t in valid_traces) / len(valid_traces)
                avg_score_excl_timeout = sum(t["score"] for t in valid_traces) / len(valid_traces)

            result = {
                "index": ex_index,
                "data_source": row.get("data_source"),
                "messages": messages,
                "rendered_prompt": prompt_text,
                "ground_truth": ground_truth,
                "avg_score": avg_score,
                "avg_acc": avg_acc,
                "avg_acc_excl_timeout": avg_acc_excl_timeout,
                "avg_score_excl_timeout": avg_score_excl_timeout,
                "num_timeouts": num_timeouts,
                "all_timed_out": all_timed_out,
                "traces": traces,
            }

            async with output_lock:
                num_correct += avg_acc_excl_timeout
                total_score += avg_score_excl_timeout
                done_examples += 1
                total_timeouts += num_timeouts
                if num_timeouts > 0:
                    examples_with_any_timeout += 1
                if output_file is not None:
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                running_acc = num_correct / done_examples if done_examples else 0.0
                print(
                    f"[{done_examples}/{num_examples}] ex{ex_index} "
                    f"data_source={row.get('data_source')} "
                    f"avg_acc_excl_timeout={avg_acc_excl_timeout:.3f} "
                    f"timeouts={num_timeouts}/{args.num_samples} "
                    f"running_acc={running_acc:.4f}",
                    flush=True,
                )

        try:
            await asyncio.gather(*[_process_example(idx, row) for idx, row in rows])
        finally:
            if output_file is not None:
                output_file.close()

        # Per-data-source breakdown.
        per_source: dict[str, dict[str, float]] = {}
        if output_path is not None and output_path.exists():
            with output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    src = rec.get("data_source") or "unknown"
                    bucket = per_source.setdefault(src, {"n": 0, "sum_acc": 0.0, "sum_score": 0.0})
                    bucket["n"] += 1
                    bucket["sum_acc"] += float(rec.get("avg_acc_excl_timeout", 0.0) or 0.0)
                    bucket["sum_score"] += float(rec.get("avg_score_excl_timeout", 0.0) or 0.0)
        per_source_summary = {
            src: {
                "num_examples": int(b["n"]),
                "accuracy": b["sum_acc"] / b["n"] if b["n"] else 0.0,
                "average_score": b["sum_score"] / b["n"] if b["n"] else 0.0,
            }
            for src, b in per_source.items()
        }

        total_samples = num_examples * args.num_samples
        return {
            "num_examples": num_examples,
            "accuracy": num_correct / num_examples if num_examples else 0.0,
            "average_score": total_score / num_examples if num_examples else 0.0,
            "per_data_source": per_source_summary,
            "timeout_stats": {
                "total_samples": total_samples,
                "total_timeouts": total_timeouts,
                "timeout_rate": total_timeouts / total_samples if total_samples else 0.0,
                "examples_with_any_timeout": examples_with_any_timeout,
                "sample_timeout_seconds": args.sample_timeout,
                "request_timeout_seconds": args.request_timeout,
            },
            "model_path": args.model_path,
            "dataset": args.dataset,
            "split": args.split,
            "num_samples": args.num_samples,
            "search_backend": args.search_backend,
            "search_url": args.search_url,
            "topk": args.topk,
            "max_turns": args.max_turns,
        }

    summary = asyncio.run(_run_eval())
    print(json.dumps(summary, indent=2))
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
