#!/usr/bin/env python3

import argparse
import glob
import importlib.util
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from prettytable import PrettyTable
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
OPENRESEARCHER_EVAL_PATH = REPO_ROOT / "third_party" / "OpenResearcher" / "eval.py"


def load_openresearcher_eval_module():
    spec = importlib.util.spec_from_file_location("openresearcher_eval", OPENRESEARCHER_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load OpenResearcher eval module from {OPENRESEARCHER_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


openresearcher_eval = load_openresearcher_eval_module()


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_") or "model"


def _record_signature(item):
    return json.dumps(item, sort_keys=True, ensure_ascii=False)


def dedupe_records_by_qid(data):
    deduped = {}
    duplicate_count = 0
    conflicting_qids = []

    for item in data:
        qid = item["qid"]
        existing = deduped.get(qid)
        if existing is None:
            deduped[qid] = item
            continue

        duplicate_count += 1

        if _record_signature(existing) == _record_signature(item):
            continue

        existing_status = existing.get("status")
        current_status = item.get("status")
        if existing_status != "success" and current_status == "success":
            deduped[qid] = item
            continue
        if existing_status == "success" and current_status != "success":
            continue

        conflicting_qids.append(qid)
        deduped[qid] = item

    return list(deduped.values()), duplicate_count, conflicting_qids


class OpenAICompatibleJudge:
    def __init__(
        self,
        llm: str,
        base_url: str,
        api_key: str,
        qps: float = 50,
        max_retries: int = 5,
        max_workers: int = 50,
    ):
        self.llm = llm
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.rate_limiter = openresearcher_eval.ThreadRateLimiter(qps)
        self.qps = qps
        self.max_workers = max_workers
        self.max_retries = max_retries

    def judge(self, data):
        output = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._judge, item) for item in data]
            for future in tqdm(as_completed(futures), total=len(futures)):
                output.append(future.result())
        return output

    def _judge(self, data):
        question = data["question"]
        content = data["messages"][-1]["content"]
        if isinstance(content, str):
            gen_output = content
        elif isinstance(content, list) and len(content) > 0:
            gen_output = content[0].get("text", "")
        else:
            gen_output = ""

        answer = data["answer"]
        prompt = openresearcher_eval.GRADER_TEMPLATE.format(
            question=question,
            response=gen_output,
            correct_answer=answer,
        )

        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.acquire()
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.llm,
                    messages=[{"role": "user", "content": prompt}],
                )
                response = openresearcher_eval.parse_judge_response(chat_completion.choices[0].message.content)
                response["qid"] = data["qid"]
                response["question"] = question
                response["gen_output"] = gen_output
                response["correct_answer"] = answer
                response["content"] = chat_completion.choices[0].message.content
                return response
            except Exception as exc:
                if attempt == self.max_retries:
                    return {"correct": False, "error": str(exc)}
                backoff = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                time.sleep(backoff)


def load_run_data(input_dir: str):
    files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    files = [path for path in files if not os.path.basename(path).startswith("evaluated")]
    files.sort()

    data = []
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            data.extend(json.loads(line) for line in handle)

    data, duplicate_count, conflicting_qids = dedupe_records_by_qid(data)
    if duplicate_count:
        print(f"Ignored {duplicate_count} duplicate records with repeated qids.")
    if conflicting_qids:
        preview = ", ".join(str(qid) for qid in conflicting_qids[:10])
        print(
            "Warning: found duplicate qids with differing payloads; "
            f"kept the last record for {len(conflicting_qids)} qids ({preview})."
        )

    clean_data = []
    error_data = []
    for item in data:
        if item.get("status") == "success":
            clean_data.append(item)
        else:
            error_data.append(item)

    return data, clean_data, error_data


def save_output(output, output_file: str):
    keys_to_remove = ["extracted_final_answer", "reasoning", "confidence", "parse_error"]
    saved_output = []
    for item in output:
        saved_item = {key: value for key, value in item.items() if key not in keys_to_remove}
        saved_output.append(saved_item)
    saved_output.sort(key=lambda item: item.get("qid", -1))

    with open(output_file, "w", encoding="utf-8") as handle:
        for item in saved_output:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_summary(data, clean_data, error_data, output):
    parsed_output = [item for item in output if not item.get("parse_error", True)]
    correct_list = [item for item in parsed_output if item.get("correct") is True]
    incorrect_list = [item for item in parsed_output if item.get("correct") is False]

    total_samples = len(data)
    success_samples = len(clean_data)
    error_samples = len(error_data)
    judged_samples = len(output)
    parsed_ok_samples = len(parsed_output)
    parse_error_samples = judged_samples - parsed_ok_samples
    correct_samples = len(correct_list)

    success_rate = (success_samples / total_samples) if total_samples > 0 else 0
    parse_error_rate = (parse_error_samples / judged_samples) if judged_samples > 0 else 0
    judged_accuracy = (correct_samples / parsed_ok_samples) if parsed_ok_samples > 0 else 0
    overall_accuracy = (correct_samples / total_samples) if total_samples > 0 else 0

    table = PrettyTable()
    table.title = "Evaluation Results Summary"
    table.field_names = ["Metric", "Count", "Percentage"]
    table.align = "l"
    table.align["Count"] = "r"
    table.align["Percentage"] = "r"

    table.add_row(["Total Samples", total_samples, f"{100:.2f}%"])
    table.add_row(["  - Success Status", success_samples, f"{success_rate:.2%}"])
    table.add_row(["  - Error Status", error_samples, f"{(1 - success_rate):.2%}"])
    table.add_row(["-" * 25, "-" * 10, "-" * 12], divider=True)
    table.add_row(["Judged Samples (Success Status)", judged_samples, f"{100:.2f}% of Success"])
    table.add_row(["  - Parsed OK", parsed_ok_samples, f"{(1 - parse_error_rate):.2%}"])
    table.add_row(["  - Parse Error", parse_error_samples, f"{parse_error_rate:.2%}"])
    table.add_row(["-" * 25, "-" * 10, "-" * 12], divider=True)
    table.add_row(["Correct Predictions", correct_samples, ""])
    table.add_row(["Judged Accuracy (Correct/Parsed OK)", "", f"{judged_accuracy:.2%}"])
    table.add_row(["Overall Accuracy (Correct/Total)", "", f"{overall_accuracy:.2%}"])

    print(table)
    return correct_list, incorrect_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_url", default="http://localhost:4141/v1")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--api_key", default="dummy")
    parser.add_argument("--qps", type=float, default=50)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=50)
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    data, clean_data, error_data = load_run_data(args.input_dir)
    print(f"Total samples: {len(data)}")
    print(f"Success samples: {len(clean_data)}")
    print(f"Error samples: {len(error_data)}")

    judge = OpenAICompatibleJudge(
        llm=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        qps=args.qps,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
    )
    output = judge.judge(clean_data)

    output_file = args.output_file
    if output_file is None:
        model_slug = sanitize_model_name(args.model)
        output_file = os.path.join(args.input_dir.rstrip("/"), f"evaluated_{model_slug}.jsonl")
    save_output(output, output_file)
    print(f"\nResults saved to: {output_file}\n")

    correct_list, incorrect_list = print_summary(data, clean_data, error_data, output)

    qid_to_data = {item["qid"]: item for item in clean_data}
    correct_turns, incorrect_turns = openresearcher_eval.collect_turn_data(correct_list, incorrect_list, qid_to_data)
    openresearcher_eval.print_turn_statistics(correct_turns, incorrect_turns)

    if correct_turns or incorrect_turns:
        openresearcher_eval.create_turn_distribution_plots(correct_turns, incorrect_turns, args.input_dir)

    correct_tool_usage, incorrect_tool_usage = openresearcher_eval.collect_tool_usage_data(
        correct_list,
        incorrect_list,
        qid_to_data,
    )
    if correct_tool_usage or incorrect_tool_usage:
        openresearcher_eval.create_tool_usage_plots(correct_tool_usage, incorrect_tool_usage, args.input_dir)


if __name__ == "__main__":
    sys.exit(main())