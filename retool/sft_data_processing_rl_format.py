from __future__ import annotations

import argparse
import json
import re
from typing import Any

from datasets import load_dataset


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use Python "
    "tools to solve mathematical problems. When you need "
    "to perform calculations, use the code_interpreter "
    "tool to execute code and get results."
)

CODE_INTERPRETER_TOOL = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": "A tool for executing Python code in a stateful Jupyter notebook. Use print() to see output.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute",
                }
            },
            "required": ["code"],
        },
    },
}

TOOL_SYSTEM_PROMPT = """# Tools

You may call one function at a time to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
__TOOLS__
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
After a tool is executed, you will receive the tool result in a user message wrapped in <tool_response></tool_response> tags.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
CODE_BLOCK_PATTERN = re.compile(r"<code>\s*```python\s*(.*?)\s*```\s*</code>", re.DOTALL)
INTERPRETER_PATTERN = re.compile(r"<interpreter>\s*(.*?)\s*</interpreter>", re.DOTALL)


def build_system_prompt() -> str:
    tool_lines = json.dumps(CODE_INTERPRETER_TOOL, ensure_ascii=False)
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{TOOL_SYSTEM_PROMPT.replace('__TOOLS__', tool_lines)}"


def stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(stringify_content(item.get("content")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "content" in content:
            return stringify_content(content.get("content"))
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def has_content(content: Any) -> bool:
    return bool(stringify_content(content).strip())


def normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
    if tool_call is None:
        return None
    if isinstance(tool_call, str):
        try:
            tool_call = json.loads(tool_call)
        except json.JSONDecodeError:
            return None
    if not isinstance(tool_call, dict):
        return None

    if "function" in tool_call and isinstance(tool_call["function"], dict):
        function = tool_call["function"]
        name = function.get("name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw_arguments": arguments}
        return {"name": name, "arguments": arguments}

    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw_arguments": arguments}
    if not name:
        return None
    return {"name": name, "arguments": arguments}


def assistant_content_from_tool_call(tool_call: dict[str, Any], prefix: str = "") -> str:
    pieces = []
    if prefix.strip():
        pieces.append(prefix.rstrip())
    pieces.append("<tool_call>")
    pieces.append(json.dumps(tool_call, ensure_ascii=False))
    pieces.append("</tool_call>")
    return "\n".join(pieces)


def keep_first_tool_call(text: str) -> str:
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return text.rstrip()
    prefix = text[: match.start()].rstrip()
    tool_call_text = match.group(1)
    try:
        tool_call = json.loads(tool_call_text.replace("\n", "\\n"))
    except json.JSONDecodeError:
        return text[: match.end()].rstrip()
    normalized_tool_call = normalize_tool_call(tool_call)
    if normalized_tool_call is None:
        return text[: match.end()].rstrip()
    return assistant_content_from_tool_call(normalized_tool_call, prefix=prefix)


def build_assistant_message(content: str = "", tool_call: dict[str, Any] | None = None) -> dict[str, str] | None:
    text = content.rstrip()
    if tool_call is not None:
        return {"role": "assistant", "content": assistant_content_from_tool_call(tool_call, prefix=text)}
    if not text:
        return None
    return {"role": "assistant", "content": text}


def split_assistant_message(message: dict[str, Any]) -> list[dict[str, str]]:
    content = stringify_content(message.get("content", ""))
    tool_calls = message.get("tool_calls") or message.get("function_call")
    converted_messages: list[dict[str, str]] = []

    if isinstance(tool_calls, list) and tool_calls:
        first_tool_call = normalize_tool_call(tool_calls[0])
        if first_tool_call is not None:
            assistant_message = build_assistant_message(content, first_tool_call)
            return [assistant_message] if assistant_message is not None else []

    if isinstance(tool_calls, dict):
        normalized = normalize_tool_call(tool_calls)
        if normalized is not None:
            assistant_message = build_assistant_message(content, normalized)
            return [assistant_message] if assistant_message is not None else []

    if "<tool_call>" in content:
        assistant_message = build_assistant_message(keep_first_tool_call(content))
        return [assistant_message] if assistant_message is not None else []

    position = 0
    while position < len(content):
        code_match = CODE_BLOCK_PATTERN.search(content, position)
        interpreter_match = INTERPRETER_PATTERN.search(content, position)

        next_match = None
        next_kind = None
        if code_match and interpreter_match:
            if code_match.start() <= interpreter_match.start():
                next_match = code_match
                next_kind = "code"
            else:
                next_match = interpreter_match
                next_kind = "interpreter"
        elif code_match:
            next_match = code_match
            next_kind = "code"
        elif interpreter_match:
            next_match = interpreter_match
            next_kind = "interpreter"

        if next_match is None:
            assistant_message = build_assistant_message(content[position:])
            if assistant_message is not None:
                converted_messages.append(assistant_message)
            break

        prefix = content[position : next_match.start()]
        if next_kind == "code":
            code = next_match.group(1).strip()
            tool_call = normalize_tool_call({"name": "code_interpreter", "arguments": {"code": code}})
            assistant_message = build_assistant_message(prefix, tool_call)
            if assistant_message is not None:
                converted_messages.append(assistant_message)
        else:
            assistant_message = build_assistant_message(prefix)
            if assistant_message is not None:
                converted_messages.append(assistant_message)
            tool_response = next_match.group(1).strip()
            if tool_response:
                converted_messages.append({"role": "user", "content": wrap_tool_response(tool_response)})

        position = next_match.end()

    return converted_messages


def wrap_tool_response(content: str) -> str:
    return f"<tool_response>\n{content.strip()}\n</tool_response>"


def convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    converted = [{"role": "system", "content": build_system_prompt()}]

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        content = stringify_content(message.get("content", ""))

        if role == "system":
            # Replace dataset-specific system prompts with the local RL-style one.
            continue
        if role == "user":
            if has_content(content):
                converted.append({"role": "user", "content": content.strip()})
            continue
        if role == "assistant":
            converted.extend(split_assistant_message(message))
            continue
        if role in {"tool", "function", "observation"}:
            if has_content(content):
                converted.append({"role": "user", "content": wrap_tool_response(content)})
            continue

        if has_content(content):
            converted.append({"role": "user", "content": content.strip()})

    return converted


def convert_sample(sample: dict[str, Any]) -> dict[str, Any]:
    raw_messages = sample.get("messages") or sample.get("conversations") or []
    return {"messages": convert_messages(raw_messages)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JoeYing/ReTool-SFT to local RL-style SFT messages.")
    parser.add_argument("--dataset", default="JoeYing/ReTool-SFT", help="Hugging Face dataset name or local path")
    parser.add_argument("--split", default="train", help="Dataset split to convert")
    parser.add_argument(
        "--output",
        default="./data/retool/ReTool-SFT-rl-format.parquet",
        help="Output parquet path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)[args.split]
    converted = dataset.map(convert_sample, remove_columns=dataset.column_names)
    converted.to_parquet(args.output)
    print(f"Saved {len(converted)} rows to {args.output}")


if __name__ == "__main__":
    main()