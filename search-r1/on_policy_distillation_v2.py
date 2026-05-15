"""Search-R1 OPD v2: task reward plus on-policy distillation.

This mirrors ReTool's v2 pattern: ``reward_func`` queries the teacher and
computes the Search-R1 EM reward, while ``post_process_rewards`` attaches
teacher log-probs and returns the non-zero Search-R1 scalar reward to GRPO.

Use with:

   --custom-rm-path on_policy_distillation_v2.reward_func
   --custom-reward-post-process-path on_policy_distillation_v2.post_process_rewards
   --use-opd --opd-type sglang --opd-kl-coef <coef>
"""

from __future__ import annotations

import sys

import torch

from slime.utils.types import Sample

from on_policy_distillation import reward_func  # noqa: F401


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Attach teacher log-probs and return Search-R1 scalar rewards."""
    teacher_log_probs = []
    n_failed = 0
    for sample in samples:
        response_length = sample.response_length
        resp = getattr(sample, "_opd_teacher_response", None)
        if resp is None:
            n_failed += 1
            teacher_log_probs.append(torch.zeros(response_length, dtype=torch.float32))
            continue
        full = torch.tensor(
            [item[0] for item in resp["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32,
        )
        sliced = full[:0] if response_length == 0 else full[-response_length:]
        if sliced.numel() != response_length:
            pad = torch.zeros(response_length - sliced.numel(), dtype=torch.float32)
            sliced = torch.cat([pad, sliced])
        teacher_log_probs.append(sliced)

    if n_failed:
        print(
            f"[search-opd-v2] WARN: {n_failed}/{len(samples)} samples had no "
            f"teacher logprobs; substituted zeros.",
            file=sys.stderr,
            flush=True,
        )

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    task_scores = [float(getattr(sample, "_opd_task_score", 0.0) or 0.0) for sample in samples]
    search_counts = [int(getattr(sample, "search_count", 0) or 0) for sample in samples]

    try:
        import wandb

        if wandb.run is not None and task_scores:
            wandb.log(
                {
                    "rollout/student_task_score": sum(task_scores) / len(task_scores),
                    "rollout/shaped_reward_mean": sum(task_scores) / len(task_scores),
                    "rollout/search_count_mean": sum(search_counts) / max(1, len(search_counts)),
                }
            )
    except ImportError:
        pass

    return task_scores, task_scores
