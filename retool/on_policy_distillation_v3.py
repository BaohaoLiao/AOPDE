"""On-policy distillation (OPD) reward functions — pure OPD with reward LOGGING.

This is a v3 variant of ``on_policy_distillation.py``. Behaviorally identical
to v1 (returns ``0.0`` reward per sample so training is *pure* distillation
driven only by the OPD reverse-KL signal), but additionally logs the math
task score, the would-be v2-style shaped reward, and the per-sample
tool-call count to wandb so you can monitor reward dynamics without using
them as the training signal.

Wire it in your launch script:
   --custom-rm-path retool.on_policy_distillation_v3.reward_func
   --custom-reward-post-process-path retool.on_policy_distillation_v3.post_process_rewards

Logged metrics (per rollout):
   rollout/student_task_score      — mean math_dapo score (+1 / -1)
   rollout/shaped_reward_mean      — mean of v2-style shaped reward (NOT used)
   rollout/tool_call_count_mean    — mean number of tool calls per sample
   rollout/student_correct_rate    — fraction of samples with task_score > 0
"""

from __future__ import annotations

import sys

import torch

from slime.utils.types import Sample

# Re-export the original async reward_func that performs the teacher /generate
# call and stashes ``_opd_teacher_response`` / ``_opd_task_score`` on the
# sample. We only override post_process_rewards.
from retool.on_policy_distillation import reward_func  # noqa: F401


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Extract teacher log-probs, log monitoring metrics, return zero rewards.

    Stores token-level teacher log-probs in ``sample.teacher_log_probs``
    (trimmed to the response span) for OPD reverse-KL computation. Returns
    ``[0.0] * N`` so the advantage estimator contributes nothing — pure OPD.
    The shaped reward and task score are logged for monitoring only.
    """
    response_lengths = [sample.response_length for sample in samples]

    # ------------------------------------------------------------------
    # 1. Teacher log-prob extraction (identical to v1).
    # ------------------------------------------------------------------
    teacher_log_probs = []
    n_failed = 0
    for sample, response_length in zip(samples, response_lengths, strict=False):
        resp = getattr(sample, "_opd_teacher_response", None)
        if resp is None:
            n_failed += 1
            teacher_log_probs.append(torch.zeros(response_length, dtype=torch.float32))
            continue
        full = torch.tensor(
            [item[0] for item in resp["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32,
        )
        # Guard against the Python slicing quirk where ``full[-0:]`` returns
        # the full tensor instead of an empty one.
        if response_length == 0:
            sliced = full[:0]
        else:
            sliced = full[-response_length:]
        if sliced.numel() != response_length:
            pad = torch.zeros(response_length - sliced.numel(), dtype=torch.float32)
            sliced = torch.cat([pad, sliced])
        teacher_log_probs.append(sliced)

    if n_failed:
        print(
            f"[opd-v3] WARN: {n_failed}/{len(samples)} samples had no teacher "
            f"logprobs; substituted zeros.",
            file=sys.stderr,
            flush=True,
        )

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # ------------------------------------------------------------------
    # 2. Compute monitoring metrics (NOT used as reward).
    # ------------------------------------------------------------------
    task_scores: list[float] = []
    tool_call_counts: list[int] = []
    shaped_rewards: list[float] = []
    for sample in samples:
        task_score = float(getattr(sample, "_opd_task_score", 0.0) or 0.0)
        num_turns = int(getattr(sample, "tool_call_count", 0) or 0)
        task_scores.append(task_score)
        tool_call_counts.append(num_turns)

        # v2-style shaped reward, computed for logging only.
        shaped = task_score
        if shaped < 0:
            tool_call_reward = (num_turns - 2) / 2 * 0.1
            shaped = min(-0.6, shaped + tool_call_reward)
        shaped_rewards.append(shaped)

    n = max(1, len(samples))
    correct_rate = sum(1 for s in task_scores if s > 0) / n

    try:
        import wandb
        if wandb.run is not None:
            wandb.log(
                {
                    "rollout/student_task_score": sum(task_scores) / n,
                    "rollout/shaped_reward_mean": sum(shaped_rewards) / n,
                    "rollout/tool_call_count_mean": sum(tool_call_counts) / n,
                    "rollout/student_correct_rate": correct_rate,
                }
            )
    except ImportError:
        pass

    print(
        f"[opd-v3] task_score_mean={sum(task_scores) / n:.3f} "
        f"correct_rate={correct_rate:.3f} "
        f"tool_calls_mean={sum(tool_call_counts) / n:.2f} "
        f"shaped_mean={sum(shaped_rewards) / n:.3f}",
        flush=True,
    )

    # Pure OPD: zero scalar reward — only the reverse-KL signal trains.
    zero_rewards = [0.0] * len(samples)
    return zero_rewards, zero_rewards
