from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import socket
import select
import subprocess
import sys
import time
import importlib
import urllib.parse
from pathlib import Path
from typing import Any
from urllib import request

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIME_ROOT = REPO_ROOT / "third_party" / "slime"
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch or reuse an SGLang server and evaluate an HF model on zhuzilin/aime-2024.")
    parser.add_argument("-n", "--num-samples", type=int, default=1, help="Number of traces to sample per prompt")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum total tokens per trace, counting both model generations and tool observations",
    )
    parser.add_argument(
        "--print-turns",
        action="store_true",
        help="Print intermediate assistant/tool turns for each sampled trace",
    )
    parser.add_argument("--model-path", required=True, help="HF model path for SGLang deployment")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path. Defaults to --model-path.",
    )
    parser.add_argument("--dataset", default="baohao/aime24", help="HF dataset name or local dataset path")
    parser.add_argument("--split", default="train", help="Dataset split to evaluate")
    parser.add_argument("--host", default="127.0.0.1", help="SGLang server host")
    parser.add_argument("--port", type=int, default=30000, help="SGLang server port")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for SGLang")
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.7,
        help="SGLang static memory fraction",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch SGLang",
    )
    parser.add_argument(
        "--extra-server-args",
        default="",
        help="Extra arguments passed to `python -m sglang.launch_server`.",
    )
    parser.add_argument(
        "--reuse-existing-server",
        action="store_true",
        help="Reuse an already-running SGLang server instead of launching one.",
    )
    parser.add_argument("--server-start-timeout", type=int, default=180, help="Seconds to wait for server startup")
    parser.add_argument("--request-timeout", type=int, default=600, help="HTTP timeout per generation request")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Max new tokens per sample")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples")
    parser.add_argument("--output", default=None, help="Optional JSONL output path for per-example results")
    parser.add_argument("--summary-output", default=None, help="Optional JSON output path for the final summary")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Max concurrent in-flight requests to the SGLang server (default: 4)",
    )
    parser.add_argument(
        "--sample-timeout",
        type=int,
        default=300,
        help="Max seconds allowed for a single trace (all turns combined). Timed-out traces are scored as incorrect (default: 300).",
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("-n/--num-samples must be at least 1")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    return args


def _load_dataset_records(args: argparse.Namespace):
    if Path(args.dataset).exists():
        dataset_path = Path(args.dataset)
        if dataset_path.suffix == ".jsonl":
            dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        elif dataset_path.suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(dataset_path), split="train")
        else:
            dataset = load_dataset(str(dataset_path), split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    if args.limit is not None:
        limit = min(args.limit, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


def _prompt_to_text(prompt: str | list[dict[str, Any]]) -> str:
    if isinstance(prompt, str):
        return prompt

    lines = []
    for message in prompt:
        if not isinstance(message, dict):
            lines.append(str(message))
            continue
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                else:
                    text_parts.append(str(item))
            content = "\n".join(part for part in text_parts if part)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _load_tokenizer(tokenizer_path: str):
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "transformers is required to run retool/eval.py. Install it in the active environment first."
        ) from exc
    return transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _load_math_dapo_compute_score():
    try:
        module = importlib.import_module("slime.rollout.rm_hub.math_dapo_utils")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import Slime math_dapo_utils from {SLIME_ROOT}. Check that the repo checkout is complete."
        ) from exc
    return module.compute_score


def _load_retool_runtime():
    try:
        return importlib.import_module("eval_generate_with_retool")
    except ImportError as exc:
        raise ImportError(
            "Failed to import retool/eval_generate_with_retool.py. Check the ReTool dependencies in the active environment."
        ) from exc


def _render_input_ids(tokenizer: Any, prompt: Any) -> list[int]:
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )


async def _post_generate(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/generate"
    return await _post_json_async(url, payload, args.request_timeout)


async def _post_json_async(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Native-async HTTP POST using asyncio streams.

    Unlike asyncio.to_thread(_post_json, ...), this is truly cancellable:
    when asyncio.wait_for times out and raises CancelledError, the TCP
    connection is closed immediately and no thread pool slot is held.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    body = json.dumps(payload).encode("utf-8")
    http_request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode() + body

    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=10)
    try:
        writer.write(http_request)
        await asyncio.wait_for(writer.drain(), timeout=10)
        # read(-1) reads until EOF — server closes connection after response with Connection: close
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
    return json.loads(response_bytes[sep + 4:])


def _truncate_text_to_token_budget(tokenizer: Any, text: str, remaining_tokens: int) -> tuple[str, int, bool]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= remaining_tokens:
        return text, len(token_ids), False
    if remaining_tokens <= 0:
        return "", 0, True
    truncated_ids = token_ids[:remaining_tokens]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=False)
    return truncated_text, len(truncated_ids), True


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _read_process_output(process: subprocess.Popen[str]) -> str:
    if process.stdout is None:
        return ""

    chunks: list[str] = []
    while True:
        ready, _, _ = select.select([process.stdout], [], [], 0)
        if not ready:
            break
        line = process.stdout.readline()
        if line == "":
            break
        chunks.append(line)
    return "".join(chunks)


def _wait_for_port(host: str, port: int, timeout: int, process: subprocess.Popen[str] | None = None) -> None:
    deadline = time.time() + timeout
    last_error = None
    startup_output: list[str] = []
    while time.time() < deadline:
        if process is not None:
            output_chunk = _read_process_output(process)
            if output_chunk:
                startup_output.append(output_chunk)
            return_code = process.poll()
            if return_code is not None:
                output_chunk = _read_process_output(process)
                if output_chunk:
                    startup_output.append(output_chunk)
                combined_output = "".join(startup_output).strip()
                detail = f"\nServer output:\n{combined_output}" if combined_output else ""
                raise RuntimeError(f"SGLang server exited during startup with code {return_code}.{detail}")
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1)
    combined_output = "".join(startup_output).strip()
    detail = f"\nServer output:\n{combined_output}" if combined_output else ""
    raise TimeoutError(f"Timed out waiting for SGLang server on {host}:{port}: {last_error}{detail}")


def _launch_server(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    if args.reuse_existing_server:
        return None

    command = [
        args.python_executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp-size",
        str(args.tp_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--trust-remote-code",
    ]
    command.extend(shlex.split(args.extra_server_args))

    env = os.environ.copy()
    print(f"Launching SGLang server: {' '.join(shlex.quote(part) for part in command)}")
    return subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _generate_one(args: argparse.Namespace, tokenizer: Any, prompt: Any) -> str:
    input_ids = _render_input_ids(tokenizer, prompt)
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    url = f"http://{args.host}:{args.port}/generate"
    output = _post_json(url, payload, timeout=args.request_timeout)
    return output["text"]


async def _generate_one_with_tools(args: argparse.Namespace, tokenizer: Any, prompt: Any, retool_runtime: Any) -> dict[str, Any]:
    tool_registry = retool_runtime.ToolRegistry()
    try:
        tool_specs = tool_registry.get_tool_specs()
        interaction_messages: list[dict[str, Any]] = []
        response_parts: list[str] = []
        turns: list[dict[str, Any]] = []
        tool_call_count = 0
        total_trace_tokens = 0
        stopped_due_to_max_tokens = False
        max_turns = retool_runtime.TOOL_CONFIGS["max_turns"]
        max_tool_calls = retool_runtime.TOOL_CONFIGS["max_tool_calls"]

        for turn_index in range(max_turns):
            if args.max_tokens is not None and total_trace_tokens >= args.max_tokens:
                stopped_due_to_max_tokens = True
                break

            rendered_prompt = retool_runtime.format_conversation_with_tools(
                prompt=prompt,
                tools=tool_specs,
                messages=interaction_messages,
            )
            input_ids = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
            payload = {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                },
            }
            output = await _post_generate(args, payload)
            cur_response = retool_runtime.postprocess_responses(output["text"])
            if args.max_tokens is not None:
                remaining_tokens = args.max_tokens - total_trace_tokens
                cur_response, response_token_count, response_was_truncated = _truncate_text_to_token_budget(
                    tokenizer,
                    cur_response,
                    remaining_tokens,
                )
            else:
                response_token_count = len(tokenizer(cur_response, add_special_tokens=False)["input_ids"])
                response_was_truncated = False

            response_parts.append(cur_response)
            total_trace_tokens += response_token_count
            turn_record = {
                "turn_index": turn_index,
                "assistant": cur_response,
            }

            if response_was_truncated:
                turn_record["max_tokens_reached"] = True
                turns.append(turn_record)
                stopped_due_to_max_tokens = True
                break

            assistant_message = retool_runtime._build_assistant_message(cur_response)
            if assistant_message is not None:
                interaction_messages.append(assistant_message)

            next_obs, done, tool_message = await retool_runtime.execute_predictions(cur_response, tool_registry)
            turn_record["done"] = done
            if done:
                turns.append(turn_record)
                break

            if "<tool_response>" in next_obs:
                tool_call_count += 1

            if args.max_tokens is not None:
                remaining_tokens = args.max_tokens - total_trace_tokens
                next_obs, observation_token_count, observation_was_truncated = _truncate_text_to_token_budget(
                    tokenizer,
                    next_obs,
                    remaining_tokens,
                )
            else:
                observation_token_count = len(tokenizer(next_obs, add_special_tokens=False)["input_ids"])
                observation_was_truncated = False

            response_parts.append(next_obs)
            total_trace_tokens += observation_token_count
            turn_record["tool_observation"] = next_obs
            if tool_message is not None:
                interaction_messages.append(tool_message)
                turn_record["tool_message"] = tool_message.get("content", "")

            if observation_was_truncated:
                turn_record["max_tokens_reached"] = True
                turns.append(turn_record)
                stopped_due_to_max_tokens = True
                break

            turns.append(turn_record)

            if tool_call_count >= max_tool_calls:
                break

        return {
            "response": "".join(response_parts),
            "turns": turns,
            "tool_call_count": tool_call_count,
            "tool_backend": tool_registry.backend,
            "sandbox_session_id": tool_registry.session_id,
            "total_trace_tokens": total_trace_tokens,
            "stopped_due_to_max_tokens": stopped_due_to_max_tokens,
        }
    finally:
        await tool_registry.close()


def _score_response(math_dapo_compute_score: Any, prompt: Any, label: str, response: str) -> dict[str, Any]:
    score_result = math_dapo_compute_score(
        _prompt_to_text(prompt) + response,
        label,
        strict_box_verify=True,
    )
    return {
        "pred": score_result.get("pred"),
        "score": float(score_result["score"]),
        "acc": bool(score_result["acc"]),
        "response": response,
    }


def _print_trace_turns(example_index: int, trace: dict[str, Any]) -> None:
    print(f"\n=== example {example_index} trace {trace['trace_index']} ===")
    print(
        f"backend={trace['tool_backend']} session={trace['sandbox_session_id']} "
        f"tool_calls={trace['tool_call_count']} total_tokens={trace['total_trace_tokens']} "
        f"score={trace['score']:.3f} acc={int(trace['acc'])}"
    )
    for turn in trace.get("turns", []):
        print(f"\n--- turn {turn['turn_index']} assistant ---")
        print(turn.get("assistant", ""))
        if "tool_message" in turn:
            print("\n--- tool message ---")
            print(turn["tool_message"])
        elif "tool_observation" in turn:
            print("\n--- tool observation ---")
            print(turn["tool_observation"])


def main() -> None:
    args = parse_args()
    tokenizer = _load_tokenizer(args.tokenizer_path or args.model_path)
    math_dapo_compute_score = _load_math_dapo_compute_score()
    retool_runtime = _load_retool_runtime()
    dataset = _load_dataset_records(args)
    output_path = Path(args.output) if args.output else None

    server_process = _launch_server(args)
    try:
        if server_process is None:
            print(
                f"Reusing existing SGLang server at http://{args.host}:{args.port}; "
                f"waiting up to {args.server_start_timeout}s for it to accept connections."
            )
        else:
            print(
                f"Waiting for launched SGLang server at http://{args.host}:{args.port} "
                f"for up to {args.server_start_timeout}s."
            )
        _wait_for_port(args.host, args.port, timeout=args.server_start_timeout, process=server_process)
        print(f"SGLang server is reachable at http://{args.host}:{args.port}")

        async def _run_eval() -> dict[str, Any]:
            num_examples = len(dataset)
            num_correct = 0
            total_score = 0.0
            # One semaphore shared across all examples for the lifetime of the
            # single event loop — avoids the broken-semaphore issue that occurs
            # when asyncio.run() is called once per example.
            semaphore = asyncio.Semaphore(args.max_concurrent)
            output_file = output_path.open("w", encoding="utf-8") if output_path else None

            try:
                for index, row in enumerate(dataset):
                    prompt = row["problem"]
                    label = str(row.get("gt", ""))

                    async def _safe_generate(idx: int) -> dict[str, Any]:
                        async with semaphore:
                            try:
                                return await asyncio.wait_for(
                                    _generate_one_with_tools(args, tokenizer, prompt, retool_runtime),
                                    timeout=args.sample_timeout,
                                )
                            except asyncio.TimeoutError:
                                print(f"[example {index} sample {idx}] timed out after {args.sample_timeout}s")
                                return {
                                    "response": "",
                                    "turns": [],
                                    "tool_call_count": 0,
                                    "tool_backend": "unknown",
                                    "sandbox_session_id": "timeout",
                                    "total_trace_tokens": 0,
                                    "stopped_due_to_max_tokens": False,
                                }

                    generations = list(await asyncio.gather(*[
                        _safe_generate(i)
                        for i in range(args.num_samples)
                    ]))

                    traces = []
                    for trace_index, generation in enumerate(generations):
                        trace = _score_response(math_dapo_compute_score, prompt, label, generation["response"])
                        trace["trace_index"] = trace_index
                        trace["turns"] = generation["turns"]
                        trace["tool_call_count"] = generation["tool_call_count"]
                        trace["tool_backend"] = generation["tool_backend"]
                        trace["sandbox_session_id"] = generation["sandbox_session_id"]
                        trace["total_trace_tokens"] = generation["total_trace_tokens"]
                        trace["stopped_due_to_max_tokens"] = generation["stopped_due_to_max_tokens"]
                        traces.append(trace)
                        if args.print_turns:
                            _print_trace_turns(index, trace)

                    avg_score = sum(t["score"] for t in traces) / len(traces)
                    avg_acc = sum(int(t["acc"]) for t in traces) / len(traces)
                    num_correct += avg_acc
                    total_score += avg_score

                    result = {
                        "index": index,
                        "label": label,
                        "avg_score": avg_score,
                        "avg_acc": avg_acc,
                        "traces": traces,
                    }

                    if output_file is not None:
                        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

                    running_acc = num_correct / (index + 1)
                    print(
                        f"[{index + 1}/{num_examples}] avg_score={avg_score:.3f} "
                        f"avg_acc={avg_acc:.3f} n={args.num_samples} running_acc={running_acc:.4f}"
                    )
            finally:
                if output_file is not None:
                    output_file.close()

            return {
                "num_examples": num_examples,
                "accuracy": num_correct / num_examples if num_examples else 0.0,
                "average_score": total_score / num_examples if num_examples else 0.0,
                "model_path": args.model_path,
                "dataset": args.dataset,
                "split": args.split,
                "num_samples": args.num_samples,
            }

        summary = asyncio.run(_run_eval())
        print(json.dumps(summary, indent=2))
        if args.summary_output:
            summary_path = Path(args.summary_output)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            print(f"Summary saved to {summary_path}")
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_process.kill()


if __name__ == "__main__":
    main()