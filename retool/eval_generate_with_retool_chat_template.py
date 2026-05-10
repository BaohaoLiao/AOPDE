# Adapted from eval_generate_with_retool.py.
# This variant renders all chat content through `tokenizer.apply_chat_template`
# instead of hand-built ChatML strings, so prompts follow whatever chat template
# (and tool-injection convention) the model's tokenizer ships with.
import json
import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

# Import tool sandbox functionality
from eval_tool_sandbox import SEMAPHORE, TOOL_CONFIGS, ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use Python "
    "tools to solve mathematical problems. When you need "
    "to perform calculations, use the code_interpreter "
    "tool to execute code and get results."
)


# ---------------------------------------------------------------------------
# Chat-template based rendering
# ---------------------------------------------------------------------------


def _build_chat_messages(
    prompt: str | list[dict[str, Any]],
    system_prompt: str = None,
    messages: list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Assemble structured chat messages used by the tokenizer chat template."""
    chat_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}
    ]
    chat_messages.extend(_normalize_prompt_messages(prompt))
    if messages:
        chat_messages.extend(
            message for message in messages if _should_keep_message(message)
        )
    return chat_messages


def format_conversation_with_tools(
    tokenizer,
    prompt: str | list[dict[str, Any]],
    tools: list[dict[str, Any]] = None,
    system_prompt: str = None,
    messages: list[dict[str, Any]] = None,
) -> str:
    """Format conversation via tokenizer.apply_chat_template (tools handled by template)."""
    chat_messages = _build_chat_messages(prompt, system_prompt=system_prompt, messages=messages)
    return tokenizer.apply_chat_template(
        chat_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def _render_recorded_messages(
    tokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] = None,
) -> str:
    """Render recorded chat messages into transcript text via the tokenizer chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )


def _render_tool_message_delta(
    tokenizer,
    prior_messages: list[dict[str, Any]],
    tool_message: dict[str, Any],
    tools: list[dict[str, Any]] = None,
    raw_assistant_text: str | None = None,
) -> str:
    """Compute the text suffix produced by appending a tool message after the last assistant turn.

    Some chat templates (notably Qwen3) render the trailing assistant turn
    differently when it is the last message vs when it is followed by a tool
    message (e.g. they inject an empty <think></think> block in one case). To
    keep `base` a true prefix of `full`, the trailing assistant turn is
    rewritten as a plain {"role": "assistant", "content": raw_assistant_text}
    on both sides so they agree byte-for-byte.
    """
    msgs_for_render = list(prior_messages)
    if (
        raw_assistant_text is not None
        and msgs_for_render
        and msgs_for_render[-1].get("role") == "assistant"
    ):
        msgs_for_render[-1] = {"role": "assistant", "content": raw_assistant_text}

    base = tokenizer.apply_chat_template(
        msgs_for_render,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    full = tokenizer.apply_chat_template(
        msgs_for_render + [tool_message],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    if full.startswith(base):
        return full[len(base):]
    # Longest common prefix fallback so we never slice mid-word if the template
    # still produces a slightly different rendering of earlier turns.
    common = 0
    max_common = min(len(base), len(full))
    while common < max_common and base[common] == full[common]:
        common += 1
    return full[common:]


# ---------------------------------------------------------------------------
# Message helpers (unchanged from the manual-render version)
# ---------------------------------------------------------------------------


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


def _has_message_content(content: Any) -> bool:
    """Return whether a message content should be kept in the rendered chat."""
    if isinstance(content, str):
        return bool(content.strip())
    return bool(_stringify_message_content(content).strip())


def _should_keep_message(message: Any) -> bool:
    """Return whether a structured message should be kept in the chat history."""
    if not isinstance(message, dict):
        return False
    if message.get("tool_calls"):
        return True
    return _has_message_content(message.get("content", ""))


def _extract_first_tool_call(prediction: str) -> dict[str, Any] | None:
    """Extract the first tool call from a model response."""
    tool_call_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if not tool_call_match:
        return None

    try:
        json_str = tool_call_match.group(1).replace("\n", "\\n")
        tool_call_data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(tool_call_data, dict):
        return None
    return tool_call_data


def _build_assistant_message(prediction: str) -> dict[str, Any] | None:
    """Build a structured assistant message from a model response."""
    sanitized_prediction = postprocess_responses(prediction)
    tool_call = _extract_first_tool_call(sanitized_prediction)
    if tool_call is None:
        content = sanitized_prediction.rstrip()
        if not content:
            return None
        return {"role": "assistant", "content": content}

    tool_call_pattern = r"<tool_call>\s*.*?\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, sanitized_prediction, re.DOTALL)
    if tool_call_match is None:
        return {"role": "assistant", "content": sanitized_prediction.rstrip()}

    content = sanitized_prediction[: tool_call_match.start()].rstrip()
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": tool_call}],
    }


def _build_initial_recorded_messages(
    prompt: str | list[dict[str, Any]], system_prompt: str = None
) -> list[dict[str, Any]]:
    """Build the initial structured message list used for debugging and replay."""
    return _build_chat_messages(prompt, system_prompt=system_prompt)


def _trim_response_token_prefix(
    tokenizer,
    raw_token_ids: list[int],
    raw_log_probs: list[float],
    sanitized_response: str,
    *,
    sandbox_session_id: str | None = None,
    turn: int | None = None,
) -> tuple[list[int], list[float]]:
    """Trim generated token/logprob sequences to the sanitized response prefix."""
    raw_response = tokenizer.decode(raw_token_ids)
    if sanitized_response == raw_response:
        return raw_token_ids, raw_log_probs

    if not sanitized_response:
        return [], []

    for prefix_len in range(1, len(raw_token_ids) + 1):
        if tokenizer.decode(raw_token_ids[:prefix_len]) == sanitized_response:
            return raw_token_ids[:prefix_len], raw_log_probs[:prefix_len]

    sanitized_token_ids = tokenizer(sanitized_response, add_special_tokens=False)["input_ids"]
    raw_suffix = raw_response[-200:].replace("\n", "\\n")
    sanitized_suffix = sanitized_response[-200:].replace("\n", "\\n")
    print(
        "[token-trim-mismatch] "
        f"session={sandbox_session_id} turn={turn} "
        f"raw_tokens={len(raw_token_ids)} sanitized_tokens={len(sanitized_token_ids)} "
        f"raw_chars={len(raw_response)} sanitized_chars={len(sanitized_response)} "
        f"raw_suffix={raw_suffix!r} sanitized_suffix={sanitized_suffix!r}"
    )
    trimmed_log_probs = raw_log_probs[: len(sanitized_token_ids)]
    if len(trimmed_log_probs) < len(sanitized_token_ids):
        trimmed_log_probs = trimmed_log_probs + [0.0] * (len(sanitized_token_ids) - len(trimmed_log_probs))
    return sanitized_token_ids, trimmed_log_probs


def _apply_final_token_clip(
    tokenizer,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    loss_masks: list[int],
    rollout_log_probs: list[float] | None,
    max_context_length: int,
) -> tuple[list[int], str, list[int], list[int], list[float] | None, bool]:
    """Hard-clip the final sample to the max token budget used by training."""
    max_response_tokens = max(0, max_context_length - len(prompt_token_ids))
    was_clipped = len(response_token_ids) > max_response_tokens

    if not was_clipped:
        response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        return prompt_token_ids + response_token_ids, response_text, response_token_ids, loss_masks, rollout_log_probs, False

    clipped_response_token_ids = response_token_ids[:max_response_tokens]
    clipped_loss_masks = loss_masks[:max_response_tokens]
    clipped_log_probs = rollout_log_probs[:max_response_tokens] if rollout_log_probs is not None else None
    clipped_response = tokenizer.decode(clipped_response_token_ids, skip_special_tokens=False)
    clipped_tokens = prompt_token_ids + clipped_response_token_ids
    return (
        clipped_tokens,
        clipped_response,
        clipped_response_token_ids,
        clipped_loss_masks,
        clipped_log_probs,
        True,
    )


def _normalize_prompt_messages(prompt: str | list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Normalize prompt input into chat messages."""
    if prompt is None:
        return []
    if isinstance(prompt, str) and "<|im_start|>" in prompt:
        parsed_messages = _parse_chat_transcript(prompt)
        if parsed_messages:
            return parsed_messages
    if isinstance(prompt, list):
        normalized_messages = []
        for message in prompt:
            if not isinstance(message, dict):
                content = str(message)
                if not content.strip():
                    continue
                normalized_messages.append({"role": "user", "content": content})
                continue
            content = _stringify_message_content(message.get("content", ""))
            if not content.strip():
                continue
            normalized_messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": content,
                }
            )
        return normalized_messages
    content = str(prompt)
    if not content.strip():
        return []
    return [{"role": "user", "content": content}]


def _parse_chat_transcript(prompt: str) -> list[dict[str, str]]:
    """Parse a rendered chat transcript back into structured messages."""
    pattern = re.compile(r"<\|im_start\|>(system|user|assistant)\n?(.*?)(?=<\|im_end\|>)<\|im_end\|>", re.DOTALL)
    messages: list[dict[str, str]] = []

    for role, content in pattern.findall(prompt):
        text = content.strip()
        if role == "assistant" and not text:
            # Ignore generation prompts like <|im_start|>assistant\n with no content.
            continue
        if not text:
            continue
        messages.append({"role": role, "content": text})

    return messages


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
    boxed_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    boxed_matches = list(re.finditer(boxed_pattern, prediction, re.DOTALL))
    answer_match = boxed_matches[-1] if boxed_matches else None
    if answer_match:
        content = answer_match.group(1).strip()
        return "answer", content

    tool_call_data = _extract_first_tool_call(prediction)
    if tool_call_data:
        tool_name = tool_call_data.get("name")
        arguments = tool_call_data.get("arguments", {})

        if tool_name == "code_interpreter":
            code = arguments.get("code", "")
            if code.strip():
                return "code", code

    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, prediction, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        return "code", content

    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
    if python_code_match:
        content = python_code_match.group(1).strip()
        return "code", content

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    if "<tool_call>" in resp:
        tool_call_pattern = r"<tool_call>\s*.*?\s*</tool_call>"
        match = re.search(tool_call_pattern, resp, re.DOTALL)
        if match:
            return resp[: match.end()]

    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"

    if "```python" in resp:
        python_pattern = r"```python\s*.*?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    if "\\boxed{" in resp:
        boxed_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(boxed_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    return resp


async def execute_predictions(
    prediction: str,
    tool_registry: ToolRegistry,
    *,
    tokenizer,
    prior_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] = None,
    raw_assistant_text: str | None = None,
) -> tuple[str, bool, dict[str, Any] | None]:
    """Execute predictions and return (next_obs_text, done, tool_message)."""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        code = content.strip()
        if code:
            result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            tool_message = {"role": "tool", "content": str(result)}
            next_obs = _render_tool_message_delta(
                tokenizer, prior_messages, tool_message, tools=tools,
                raw_assistant_text=raw_assistant_text,
            )
            done = False
        else:
            tool_message = {"role": "tool", "content": "Error: No Python code found"}
            next_obs = _render_tool_message_delta(
                tokenizer, prior_messages, tool_message, tools=tools,
                raw_assistant_text=raw_assistant_text,
            )
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
        tool_message = None
    else:
        tool_message = {
            "role": "tool",
            "content": (
                "The previous action is invalid. "
                "If executing code, you should return a JSON object inside <tool_call></tool_call>. "
                "If giving the final answer, you should use the format 'Answer: \\boxed{answer}'. PLease try again."
            ),
        }
        next_obs = _render_tool_message_delta(
            tokenizer, prior_messages, tool_message, tools=tools,
            raw_assistant_text=raw_assistant_text,
        )
        done = False

    return next_obs, done, tool_message


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls (chat-template variant)."""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    tool_registry = ToolRegistry()
    sandbox_session_id = tool_registry.session_id
    sandbox_backend = tool_registry.backend
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Initial prompt: system + user, with tool schemas injected by the chat template.
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(
        state.tokenizer, prompt=sample.prompt, tools=tool_specs
    )
    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # Tool descriptions are injected by the chat template; the recorded system
    # message only stores the base instruction.
    recorded_messages = _build_initial_recorded_messages(sample.prompt)
    interaction_messages: list[dict[str, Any]] = []
    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu

    max_context_length = max_context_length - 16  # buffer

    sample.tokens = list(prompt_tokens_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.sandbox_session_id = sandbox_session_id
    sample.sandbox_backend = sandbox_backend
    sample.messages = list(recorded_messages)

    print(f"[sandbox] session={sandbox_session_id} backend={sandbox_backend} sample_start")

    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0
    last_finish_reason = None

    for turn in range(TOOL_CONFIGS["max_turns"]):
        prompt = format_conversation_with_tools(
            state.tokenizer,
            prompt=sample.prompt,
            tools=tool_specs,
            messages=interaction_messages,
        )
        current_prompt_token_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        total_length = len(current_prompt_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break
        remaining_context = max_context_length - total_length

        current_token_ids = current_prompt_token_ids
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
            "return_logprob": True,
        }

        print(
            f"[sandbox] session={sandbox_session_id} backend={sandbox_backend} turn={turn} tool_calls={tool_call_count}"
        )

        output = await post(url, payload)

        last_finish_reason = output["meta_info"]["finish_reason"]["type"]

        if last_finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            break

        if "output_token_logprobs" in output["meta_info"]:
            raw_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            raw_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            raw_response = state.tokenizer.decode(raw_response_token_ids)
            cur_response = postprocess_responses(raw_response)
            cur_response_token_ids, cur_log_probs = _trim_response_token_prefix(
                state.tokenizer,
                raw_response_token_ids,
                raw_log_probs,
                cur_response,
                sandbox_session_id=sandbox_session_id,
                turn=turn,
            )
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

        assistant_message = _build_assistant_message(cur_response)
        if assistant_message is not None:
            interaction_messages.append(assistant_message)
            recorded_messages.append(assistant_message)
            sample.messages = list(recorded_messages)

        if last_finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
            break

        # Render the tool observation as a delta against the current chat state
        # using the tokenizer's chat template.
        prior_messages_for_tool = _build_chat_messages(
            sample.prompt, messages=interaction_messages
        )
        next_obs, done, tool_message = await execute_predictions(
            cur_response,
            tool_registry,
            tokenizer=state.tokenizer,
            prior_messages=prior_messages_for_tool,
            tools=tool_specs,
            raw_assistant_text=cur_response,
        )
        if done:
            break

        if "<tool_response>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        remaining_observation_tokens = max_context_length - (
            len(current_prompt_token_ids) + len(cur_response_token_ids)
        )
        if remaining_observation_tokens <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        if len(obs_tokens_ids) > remaining_observation_tokens:
            obs_tokens_ids = obs_tokens_ids[-remaining_observation_tokens:]
            next_obs = state.tokenizer.decode(obs_tokens_ids, skip_special_tokens=False)
            sample.status = Sample.Status.TRUNCATED
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        if tool_message is not None:
            interaction_messages.append(tool_message)
            recorded_messages.append(tool_message)
            sample.messages = list(recorded_messages)

        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

            assert len(response_token_ids) == len(
                sample.rollout_log_probs
            ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"

        if sample.status == Sample.Status.TRUNCATED:
            break

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    (
        final_tokens,
        final_response,
        final_response_token_ids,
        final_loss_masks,
        final_rollout_log_probs,
        final_was_clipped,
    ) = _apply_final_token_clip(
        state.tokenizer,
        prompt_tokens_ids,
        response_token_ids,
        loss_masks,
        sample.rollout_log_probs,
        max_context_length,
    )

    sample.tokens = final_tokens
    sample.response_length = len(final_response_token_ids)
    sample.response = final_response
    sample.loss_mask = final_loss_masks
    sample.rollout_log_probs = final_rollout_log_probs
    if final_was_clipped:
        sample.status = Sample.Status.TRUNCATED

    sample.payload_text = _render_recorded_messages(
        state.tokenizer, recorded_messages, tools=tool_specs
    )
    sample.payload_has_system = any(m.get("role") == "system" for m in recorded_messages)
    sample.payload_has_tools = bool(tool_specs)
    sample.sandbox_session_id = sandbox_session_id
    sample.sandbox_backend = sandbox_backend

    sample.tool_call_count = tool_call_count

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["round_number"] = tool_call_count

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

    solution_str = _prompt_to_text(sample.prompt) + sample.response

    ground_truth = sample.label if sample.label is not None else ""

    num_turns = getattr(sample, "tool_call_count", 0)

    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
