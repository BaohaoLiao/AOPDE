"""On-policy distillation (OPD) reward functions WITH math+tool_call shaping.

This is a v2 variant of ``on_policy_distillation.py`` that returns a non-zero
scalar reward (math correctness + tool-call shaping) to the advantage
estimator IN ADDITION to the OPD reverse-KL signal. Use this when you want
mixed RL+OPD training instead of pure distillation.

Wire it in your launch script:
   --custom-rm-path retool.on_policy_distillation_v2.reward_func
   --custom-reward-post-process-path retool.on_policy_distillation_v2.post_process_rewards

The original ``on_policy_distillation.py`` is left untouched for runs that
want pure distillation (reward = 0).

Reward formula (mirrors ``generate_with_retool.reward_func``):
   - +1.0 when the math answer is correct
   - otherwise ``min(-0.6, base_negative + (num_turns - 2) / 2 * 0.1)``
   The OPD reverse-KL is applied on top by ``apply_opd_kl_to_advantages``.
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
    """Extract teacher log-probs AND return shaped scalar reward.

    Stores token-level teacher log-probs in ``sample.teacher_log_probs``
    (trimmed to the response span) for OPD reverse-KL computation. Returns
    a math + tool-call shaped reward per sample (mirrors
    ``generate_with_retool.reward_func``):
      - +1.0 when the answer is correct
      - otherwise ``min(-0.6, base_negative + (num_turns - 2)/2 * 0.1)``
    The OPD reverse-KL penalty is applied on top of this reward by
    ``apply_opd_kl_to_advantages``.

    Note: ``sample.response_length`` spans model-generated tokens **and**
    tool-observation tokens. Observation tokens have ``loss_mask=0`` so
    their KL penalty contribution is zeroed out in the final training loss.
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
            f"[opd-v2] WARN: {n_failed}/{len(samples)} samples had no teacher "
            f"logprobs; substituted zeros.",
            file=sys.stderr,
            flush=True,
        )

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # ------------------------------------------------------------------
    # 2. Math + tool-call shaping reward (NEW in v2).
    #    - +1.0 when correct
    #    - min(-0.6, base + (num_turns - 2)/2 * 0.1) when wrong
    # ------------------------------------------------------------------
    scalar_rewards: list[float] = []
    task_scores: list[float] = []
    tool_call_counts: list[int] = []
    for sample in samples:
        task_score = float(getattr(sample, "_opd_task_score", 0.0) or 0.0)
        num_turns = int(getattr(sample, "tool_call_count", 0) or 0)
        task_scores.append(task_score)
        tool_call_counts.append(num_turns)

        score = task_score
        if score < 0:
            tool_call_reward = (num_turns - 2) / 2 * 0.1
            score = min(-0.6, score + tool_call_reward)
        scalar_rewards.append(score)

    try:
        import wandb
        if wandb.run is not None:
            wandb.log(
                {
                    "rollout/student_task_score": sum(task_scores) / len(task_scores),
                    "rollout/shaped_reward_mean": sum(scalar_rewards) / len(scalar_rewards),
                    "rollout/tool_call_count_mean": sum(tool_call_counts) / max(1, len(tool_call_counts)),
                }
            )
    except ImportError:
        pass

    return scalar_rewards, scalar_rewards
