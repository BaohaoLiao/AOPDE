"""On-policy distillation reward functions for retool.

Usage (add to RM_ARGS in the training script):

    RM_ARGS=(
       --custom-rm-path retool.on_policy_distillation.reward_func
       --custom-reward-post-process-path retool.on_policy_distillation.post_process_rewards
       --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
    )

The teacher log-probs cover all response tokens (model generations + tool
observations). Tool observation positions have loss_mask=0 so the KL penalty
at those positions is zeroed out during the training loss computation.

The scalar task reward returned to the advantage estimator is always 0.0 —
the only learning signal is the reverse-KL penalty computed by
apply_opd_kl_to_advantages. The math_dapo score is computed and logged to
wandb for monitoring the student's task performance, but is NOT used as a
reward.
"""
from __future__ import annotations

import re

import aiohttp
import torch

from slime.utils.types import Sample

try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("math_dapo_utils is not installed") from e


# ---------------------------------------------------------------------------
# Helpers (mirrors generate_with_retool._prompt_to_text)
# ---------------------------------------------------------------------------

def _prompt_to_text(prompt: str | list) -> str:
    """Convert prompt data into a plain string for reward computation."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        lines = []
        for message in prompt:
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        item.get("text", "") for item in content if isinstance(item, dict)
                    )
                lines.append(f"{message.get('role', 'user')}: {content}")
            else:
                lines.append(str(message))
        return "\n".join(lines)
    return str(prompt)


# ---------------------------------------------------------------------------
# reward_func — called asynchronously once per sample during rollout
# ---------------------------------------------------------------------------

async def reward_func(args, sample: Sample, **kwargs):
    """Query teacher log-probs and compute math_dapo score for monitoring.

    Returns a dict stored in ``sample.reward`` that ``post_process_rewards``
    unpacks later.  Do NOT set ``--reward-key`` when using this function:
    ``post_process_rewards`` reads ``sample.reward`` directly.
    """
    # 1. Teacher forward pass (no generation, just log-probs)
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            teacher_response = await resp.json()

    # 2. Math score — for monitoring only, NOT used as a reward
    solution_str = _prompt_to_text(sample.prompt) + sample.response
    ground_truth = sample.label if sample.label is not None else ""
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    return {
        "teacher_response": teacher_response,
        "task_score": result["score"],
    }


# ---------------------------------------------------------------------------
# post_process_rewards — called once per batch after all reward_func calls
# ---------------------------------------------------------------------------

def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Extract teacher log-probs for OPD training and log task score for monitoring.

    Stores token-level teacher log-probs in ``sample.teacher_log_probs``
    (trimmed to the response span).  Scalar rewards returned to the advantage
    estimator are always 0.0 — the learning signal comes entirely from the
    reverse-KL penalty applied by ``apply_opd_kl_to_advantages``.

    The math_dapo task score is logged to wandb under
    ``rollout/student_task_score`` for monitoring only.

    Note: ``sample.response_length`` spans model-generated tokens **and**
    tool-observation tokens.  Observation tokens have ``loss_mask=0`` so
    their KL penalty contribution is zeroed out in the final training loss.
    """
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the SGLang response.
    # ``input_token_logprobs`` contains one entry per input token; we skip
    # the first element (the BOS / prompt-start position has no predecessor).
    teacher_log_probs = [
        torch.tensor(
            [item[0] for item in sample.reward["teacher_response"]["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32,
        )
        for sample in samples
    ]

    # Trim to the response span: last response_length tokens.
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Log task score for monitoring (not used as reward)
    task_scores = [sample.reward["task_score"] for sample in samples]
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"rollout/student_task_score": sum(task_scores) / len(task_scores)})
    except ImportError:
        pass

    # Pure distillation: return 0.0 task rewards so the advantage estimator
    # sees no task signal.  The OPD KL penalty is the only learning signal.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
