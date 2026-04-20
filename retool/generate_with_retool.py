# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import json
import re
from typing import Any

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

# Import tool sandbox functionality
from tool_sandbox import SEMAPHORE, TOOL_CONFIGS, ToolRegistry

# Jinja2 template for tool-enabled conversations
TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one function at a time to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
After a tool is executed, you will receive the tool result in a user message wrapped in <tool_response></tool_response> tags.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


def format_conversation_with_tools(
    prompt: str | list[dict[str, Any]],
    tools: list[dict[str, Any]] = None,
    system_prompt: str = None,
    messages: list[dict[str, Any]] = None,
) -> str:
    """Format conversation using Jinja2 template with tool support"""
    template = Template(TOOL_TEMPLATE)

    # Prepare messages
    messages_to_render = []

    # Always add system message - use provided one or default
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = (
            "You are a helpful assistant that can use Python "
            "tools to solve mathematical problems. When you need "
            "to perform calculations, use the code_interpreter "
            "tool to execute code and get results."
        )

    messages_to_render.append({"role": "system", "content": system_content})

    prompt_messages = _normalize_prompt_messages(prompt)
    if prompt_messages:
        messages_to_render.extend(prompt_messages)

    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(messages)

    # Render template
    formatted_text = template.render(messages=messages_to_render, tools=tools or [])

    return formatted_text


def _stringify_message_content(content: Any) -> str:
    """Convert structured message content into plain text for prompting and rewards."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(_stringify_message_content(item.get("content")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "content" in content:
            return _stringify_message_content(content.get("content"))
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _normalize_prompt_messages(prompt: str | list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Normalize prompt input into chat messages."""
    if prompt is None:
        return []
    if isinstance(prompt, list):
        normalized_messages = []
        for message in prompt:
            if not isinstance(message, dict):
                normalized_messages.append({"role": "user", "content": str(message)})
                continue
            normalized_messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": _stringify_message_content(message.get("content", "")),
                }
            )
        return normalized_messages
    return [{"role": "user", "content": str(prompt)}]


def _prompt_to_text(prompt: str | list[dict[str, Any]]) -> str:
    """Convert prompt data into a string for reward computation."""
    if isinstance(prompt, str):
        return prompt
    normalized_messages = _normalize_prompt_messages(prompt)
    lines = []
    for message in normalized_messages:
        lines.append(f"{message['role']}: {message['content']}")
    return "\n".join(lines)


def postprocess_predictions(prediction: str):
    """Extract action and content from prediction string"""
    # Check for Answer: \boxed{...} format (only format we need for math_dapo)
    # Use a more robust regex that handles nested braces
    answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        return "answer", content

    # Then check for <tool_call> tags (new format from Jinja2 template)
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            import json

            # Clean up the JSON string by removing newlines and extra
            # whitespace
            json_str = tool_call_match.group(1)
            # Replace newlines in string values with \n
            json_str = json_str.replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})

            if tool_name == "code_interpreter":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

    # Then check for <code> tags
    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, prediction, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        return "code", content

    # Finally check for ```python code blocks (lowest priority)
    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
    if python_code_match:
        content = python_code_match.group(1).strip()
        return "code", content

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    # Handle <tool_call> tags (new format from Jinja2 template)
    if "<tool_call>" in resp:
        # Keep only the first complete <tool_call>...</tool_call> block.
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        match = re.search(tool_call_pattern, resp, re.DOTALL)
        if match:
            return resp[: match.end()]

    # Handle <code> tags
    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"

    # Handle ```python code blocks
    if "```python" in resp:
        # Find the last occurrence of ```python...```
        python_pattern = r"```python\s*.*?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle Answer: \boxed{...} format (only format we need for math_dapo)
    if "Answer:" in resp and "\\boxed{" in resp:
        # Find the last occurrence of Answer: \boxed{...} with nested braces support
        answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    return resp


async def execute_predictions(prediction: str, tool_registry: ToolRegistry) -> str:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        # Content is already the Python code (extracted by
        # postprocess_predictions)
        code = content.strip()
        if code:
            async with SEMAPHORE:
                result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            next_obs = (
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "<tool_response>\n"
                f"{result}\n"
                "</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            done = False
        else:
            next_obs = (
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "<tool_response>\n"
                "Error: No Python code found\n"
                "</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "My previous action is invalid. "
            "If I want to execute code, I should return a JSON object inside <tool_call></tool_call>. "
            "If I want to give the final answer, I should use the format 'Answer: \\boxed{answer}'. Let me try again.\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."

    state = GenerateState(args)
    tool_registry = ToolRegistry()
    sandbox_session_id = tool_registry.session_id
    sandbox_backend = tool_registry.backend
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Set up the initial prompt with system prompt and tools (outside the loop)
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu

    # Keep the sample structurally valid even if a later rollout turn aborts.
    sample.tokens = list(prompt_tokens_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.sandbox_session_id = sandbox_session_id
    sample.sandbox_backend = sandbox_backend

    print(f"[sandbox] session={sandbox_session_id} backend={sandbox_backend} sample_start")

    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds
    last_finish_reason = None

    for turn in range(TOOL_CONFIGS["max_turns"]):
        # Check if total length exceeds max context length
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break
        remaining_context = max_context_length - total_length

        # Use token IDs instead of text
        current_token_ids = prompt_tokens_ids + response_token_ids
        current_sampling_params = dict(sampling_params)
        current_sampling_params["max_new_tokens"] = min(
            current_sampling_params["max_new_tokens"],
            remaining_context,
        )
        if current_sampling_params["max_new_tokens"] <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": current_sampling_params,
            "return_logprob": True,  # Request log probabilities for training
        }

        # Log payload to wandb for debugging
        try:
            import wandb

            if wandb.run is not None:
                # Count available tools (from tool_specs)
                available_tools = len(tool_specs)
                # Count tools used in the current response
                tools_used = response.count("<tool_response>")

                wandb.log(
                    {
                        "debug/payload_length": len(prompt + response),
                        "debug/available_tools": available_tools,
                        "debug/sandbox_backend": sandbox_backend,
                        "debug/sandbox_session_id": sandbox_session_id,
                        "debug/tools_used": tools_used,
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass  # wandb not available

        print(
            f"[sandbox] session={sandbox_session_id} backend={sandbox_backend} turn={turn} tool_calls={tool_call_count}"
        )

        output = await post(url, payload)

        last_finish_reason = output["meta_info"]["finish_reason"]["type"]

        # Handle abort
        if last_finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            break

        if "output_token_logprobs" in output["meta_info"]:
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs

        else:
            cur_response = output["text"]
            cur_response = postprocess_responses(cur_response)
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # Check length limit
        if last_finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
            break

        next_obs, done = await execute_predictions(cur_response, tool_registry)
        if done:
            break

        # Count tool calls (when we get interpreter output, it means a tool
        # was called)
        if "<tool_response>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        remaining_context = max_context_length - (len(prompt_tokens_ids) + len(response_token_ids))
        if remaining_context <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        if len(obs_tokens_ids) > remaining_context:
            obs_tokens_ids = obs_tokens_ids[:remaining_context]
            next_obs = state.tokenizer.decode(obs_tokens_ids, skip_special_tokens=False)
            sample.status = Sample.Status.TRUNCATED
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens (they won't be used due to loss_mask=0)
        # Check if maximum tool call count reached
        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

            assert len(response_token_ids) == len(
                sample.rollout_log_probs
            ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"

        if sample.status == Sample.Status.TRUNCATED:
            break

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # Store payload information for wandb logging
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in prompt + response
    sample.payload_has_tools = "# Tools" in prompt + response
    sample.sandbox_session_id = sandbox_session_id
    sample.sandbox_backend = sandbox_backend

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count

    # Set status
    if sample.status not in {Sample.Status.TRUNCATED, Sample.Status.ABORTED}:
        match last_finish_reason:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

    await tool_registry.close()
    return sample


async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Build complete solution string
    solution_str = _prompt_to_text(sample.prompt) + sample.response

    # Get ground truth answer - label is a string, not a dict
    ground_truth = sample.label if sample.label is not None else ""

    # Get tool call count as num_turns
    num_turns = getattr(sample, "tool_call_count", 0)

    # use \\boxed{...} answer
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
