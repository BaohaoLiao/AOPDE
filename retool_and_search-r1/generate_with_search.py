# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
# This is a unified version supporting both local search and Google search, with optional log probability collection

import asyncio
import json
import os
import re

from qa_em_format import compute_score_em

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# System prompt and tool definition, matched with eval_generate_with_search.py.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools to answer questions. "
    "You have access to a search tool. When you need to look up information, "
    "call the search tool as described below."
)

TOOL_SYSTEM_PROMPT = """# Tools

You may call one function at a time to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
<tool name=\"search\" description=\"Search for information.\" args=\"{\\\"query\\\": string}\" />
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
After a tool is executed, you will receive the tool result in a user message wrapped in <tool_response></tool_response> tags.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

# Configuration for Search-R1
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 4,
    "topk": 3,
    "search_concurrency": 256,
    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google"
    # ============== Local Search Configuration ==============
    # (Only used when search_backend="local")
    "local": {
        "search_url": os.environ.get(
            "SEARCH_URL",
            "http://127.0.0.1:8000/retrieve",
        ),  # Comma-separate multiple local retriever URLs.
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Google Search Configuration ==============
    # (Only used when search_backend="google")
    "google": {
        "api_key": "your_api_key_here",  # Replace with your actual API key
        "snippet_only": True,  # Set to True to only return snippets
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Log Probability Collection ==============
    "return_logprob": True,  # Set to True to collect log probabilities for TIS metrics
    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])


def _tool_system_content() -> str:
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{TOOL_SYSTEM_PROMPT}"


def _system_block() -> str:
    return f"<|im_start|>system\n{_tool_system_content()}<|im_end|>\n"


def format_conversation_with_tools(prompt: str) -> str:
    """Add the eval-style system/tool prompt to a raw or already-rendered prompt."""
    if TOOL_SYSTEM_PROMPT in prompt or '<tool name="search"' in prompt:
        return prompt

    if prompt.startswith("<|im_start|>system"):
        end_tag = "<|im_end|>"
        end = prompt.find(end_tag)
        if end != -1:
            system_content_end = end
            return (
                prompt[:system_content_end].rstrip()
                + "\n\n"
                + TOOL_SYSTEM_PROMPT
                + prompt[system_content_end:]
            )

    if "<|im_start|>user" in prompt:
        return _system_block() + prompt

    return (
        _system_block()
        + f"<|im_start|>user\n{prompt}<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def _extract_user_messages(prompt: str) -> list[str]:
    matches = re.findall(r"<\|im_start\|>user\s*(.*?)<\|im_end\|>", prompt, re.DOTALL)
    if matches:
        return [match.strip() for match in matches if match.strip()]
    return [prompt.strip()] if prompt.strip() else []


def _extract_tool_response_content(observation: str) -> str | None:
    match = re.search(r"<tool_response>\s*(.*?)\s*</tool_response>", observation, re.DOTALL)
    if match:
        return match.group(1).strip()
    start_tag = "<tool_response>"
    start = observation.find(start_tag)
    if start != -1:
        return observation[start + len(start_tag):].strip()
    return None


def _truncate_text_to_token_budget(tokenizer, text: str, remaining_tokens: int) -> tuple[str, list[int], bool]:
    if remaining_tokens <= 0:
        return "", [], True
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= remaining_tokens:
        return text, token_ids, False
    truncated_ids = token_ids[:remaining_tokens]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=False)
    return truncated_text, truncated_ids, True


def _apply_final_token_clip(
    tokenizer,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    loss_mask: list[int],
    rollout_log_probs: list[float] | None,
    max_context_length: int,
) -> tuple[list[int], str, list[int], list[int], list[float] | None, bool]:
    max_response_tokens = max(0, max_context_length - len(prompt_token_ids))
    if len(response_token_ids) <= max_response_tokens:
        response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        return prompt_token_ids + response_token_ids, response_text, response_token_ids, loss_mask, rollout_log_probs, False

    clipped_response_token_ids = response_token_ids[:max_response_tokens]
    clipped_loss_mask = loss_mask[:max_response_tokens]
    clipped_log_probs = rollout_log_probs[:max_response_tokens] if rollout_log_probs is not None else None
    clipped_response = tokenizer.decode(clipped_response_token_ids, skip_special_tokens=False)
    return (
        prompt_token_ids + clipped_response_token_ids,
        clipped_response,
        clipped_response_token_ids,
        clipped_loss_mask,
        clipped_log_probs,
        True,
    )


def _initial_trace_messages(raw_prompt: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": _tool_system_content()}]
    for content in _extract_user_messages(raw_prompt):
        messages.append({"role": "user", "content": content})
    return messages


def _passages2string(retrieval_result):
    """
    Convert retrieval results to a formatted string.
    This function works with both google_search and local_search results.
    """
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference


async def search(query: str) -> str:
    """
    Perform search using either local search engine or Google search.
    The search backend is determined by SEARCH_R1_CONFIGS["search_backend"].
    """
    backend = SEARCH_R1_CONFIGS["search_backend"]

    if backend == "local":
        from local_search_server import local_search

        local_config = SEARCH_R1_CONFIGS["local"]
        result = await local_search(
            local_config["search_url"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            proxy=local_config["proxy"],
        )
    elif backend == "google":
        from google_search_server import google_search

        google_config = SEARCH_R1_CONFIGS["google"]
        result = await google_search(
            google_config["api_key"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            snippet_only=google_config["snippet_only"],
            proxy=google_config["proxy"],
        )
    else:
        raise ValueError(f"Unknown search backend: {backend}. " f"Must be either 'local' or 'google'.")

    return _passages2string(result)


def extract_boxed_answer(prediction: str) -> str | None:
    """Extract the final answer from the last LaTeX-style \boxed{...}, if present."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", prediction)
    if matches:
        return matches[-1].strip()
    return None


# IMPORTANT: When we need to collect log probabilities (logp), we CANNOT change
# strings returned from SGLang because token/logp arrays would no longer align.
# Therefore, postprocess_responses is only used when return_logprob=False.
def postprocess_responses(resp: str) -> str:
    """
    Post-process response to ensure tag completeness.
    Only used when SEARCH_R1_CONFIGS["return_logprob"] is False.
    """
    if "</tool_call>" in resp:
        return resp.split("</tool_call>")[0] + "</tool_call>"
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    return resp


def _extract_first_tool_call(prediction: str) -> dict | None:
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


def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    """Return (action, content) where action is 'search'|'answer'|'boxed_answer'|None."""
    tool_call = _extract_first_tool_call(prediction)
    if tool_call and tool_call.get("name") == "search":
        arguments = tool_call.get("arguments", {})
        if isinstance(arguments, dict):
            return "search", str(arguments.get("query", ""))
        return "search", ""

    match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if match:
        return "answer", match.group(1).strip()

    boxed = extract_boxed_answer(prediction)
    if boxed is not None:
        return "boxed_answer", boxed

    return None, ""


async def execute_predictions(prediction: str) -> str:
    action, content = postprocess_predictions(prediction)

    if action == "search":
        try:
            async with SEMAPHORE:
                search_results = await search(content)
            tool_response_content = search_results.strip()
        except Exception as e:
            tool_response_content = f"[ERROR] {type(e).__name__}: {e}"
        next_obs = (
            "<|im_start|>user\n<tool_response>\n"
            f"{tool_response_content}\n"
            "</tool_response><|im_end|>\n<|im_start|>assistant\n"
        )
        done = False
    elif action == "answer" or action == "boxed_answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "<|im_start|>user\n"
            "Your previous action is invalid. "
            "If you want to use a tool, you should return a <tool_call>...</tool_call> block. "
            "If you want to give the final answer, you should put the answer in \\boxed{{}}. "
            "Please try again.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Set up the initial prompt with the same system/tool instructions used by eval.
    prompt_text = format_conversation_with_tools(sample.prompt)
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    max_context_length = args.rollout_max_response_len - 16
    max_tokens_per_gpu = getattr(args, "max_tokens_per_gpu", None)
    if max_tokens_per_gpu is not None:
        max_context_length = min(max_context_length, args.context_parallel_size * max_tokens_per_gpu - 16)
    if len(prompt_tokens_ids) >= max_context_length:
        sample.tokens = prompt_tokens_ids[:max_context_length]
        sample.response_length = 0
        sample.response = ""
        sample.loss_mask = []
        sample.prompt = state.tokenizer.decode(sample.tokens, skip_special_tokens=False)
        sample.status = Sample.Status.TRUNCATED
        return sample

    response = ""
    response_token_ids = []
    loss_mask = []
    rollout_log_probs = [] if SEARCH_R1_CONFIGS["return_logprob"] else None
    messages = _initial_trace_messages(sample.prompt)
    search_count = 0
    observation_truncated = False
    final_was_clipped = False
    last_finish_reason = None

    for _turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break
        remaining_context = max_context_length - total_length
        turn_sampling_params = dict(sampling_params)
        turn_sampling_params["stop"] = ["</answer>"]
        turn_sampling_params["max_new_tokens"] = min(
            turn_sampling_params.get("max_new_tokens", remaining_context),
            remaining_context,
        )
        if turn_sampling_params["max_new_tokens"] <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        payload = {
            "text": prompt_text + response,
            "sampling_params": turn_sampling_params,
        }
        # Add log probability collection if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)
        last_finish_reason = output["meta_info"]["finish_reason"]["type"]

        # abort
        if last_finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]

        # Extract tokens and log probs based on configuration
        if SEARCH_R1_CONFIGS["return_logprob"]:
            # Extract log probs from output - required for TIS metrics
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError(
                    "output_token_logprobs not found in output meta_info. "
                    "Make sure 'return_logprob': True is set in the payload."
                )

            # Use token IDs and log probs directly from output_token_logprobs
            # This ensures perfect alignment between tokens and log probs
            # output_token_logprobs format: [[log_prob, token_id, ...], ...]
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            # When not collecting log probs, we can safely postprocess the response
            cur_response = postprocess_responses(cur_response)
            # Tokenize the (possibly postprocessed) response
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)
        messages.append({"role": "assistant", "content": cur_response})

        # Add log probs if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += cur_response_log_probs

        if last_finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break
        tool_response_content = _extract_tool_response_content(next_obs)

        assert next_obs != "", "Next observation should not be empty."
        remaining_observation_tokens = max_context_length - (len(prompt_tokens_ids) + len(response_token_ids))
        if remaining_observation_tokens <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        next_obs, obs_tokens_ids, obs_was_truncated = _truncate_text_to_token_budget(
            state.tokenizer,
            next_obs,
            remaining_observation_tokens,
        )
        if obs_was_truncated:
            observation_truncated = True
            sample.status = Sample.Status.TRUNCATED
        if tool_response_content is not None:
            truncated_tool_response_content = _extract_tool_response_content(next_obs) or ""
            messages.append({"role": "tool", "content": truncated_tool_response_content})
            search_count += 1
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens if enabled (they won't be used due to loss_mask=0)
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += [0.0] * len(obs_tokens_ids)

            # Verify alignment when collecting log probs
            assert len(response_token_ids) == len(
                rollout_log_probs
            ), f"Token/logp length mismatch: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"
        if obs_was_truncated:
            break

    # Store statistics for wandb logging
    (
        final_tokens,
        final_response,
        final_response_token_ids,
        final_loss_mask,
        final_rollout_log_probs,
        final_was_clipped,
    ) = _apply_final_token_clip(
        state.tokenizer,
        prompt_tokens_ids,
        response_token_ids,
        loss_mask,
        rollout_log_probs,
        max_context_length,
    )

    sample.tokens = final_tokens
    sample.response_length = len(final_response_token_ids)
    sample.response = final_response
    sample.loss_mask = final_loss_mask
    sample.prompt = prompt_text
    sample.messages = messages
    sample.payload_text = prompt_text + final_response
    sample.payload_has_system = "<|im_start|>system" in sample.payload_text
    sample.payload_has_tools = "# Tools" in sample.payload_text
    sample.tool_call_count = search_count
    sample.search_count = search_count
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["messages"] = messages
    sample.metadata["payload_text"] = sample.payload_text
    sample.metadata["payload_has_system"] = sample.payload_has_system
    sample.metadata["payload_has_tools"] = sample.payload_has_tools
    sample.metadata["tool_call_count"] = search_count
    sample.metadata["search_count"] = search_count
    sample.metadata["max_context_length"] = max_context_length
    sample.metadata["observation_truncated"] = observation_truncated
    sample.metadata["final_was_clipped"] = final_was_clipped

    # Store log probs if enabled
    if SEARCH_R1_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = final_rollout_log_probs if final_rollout_log_probs else None

    if final_was_clipped:
        sample.status = Sample.Status.TRUNCATED

    if sample.status == Sample.Status.PENDING:
        match last_finish_reason:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """The reward function for retrieval-based question answering.

    Args:
        args: the arguments
        sample: the sample to evaluate
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    score = compute_score_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label["ground_truth"],
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    return score
