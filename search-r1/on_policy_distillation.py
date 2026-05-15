"""On-policy distillation reward helpers for Search-R1.

This module is intentionally separate from ``generate_with_search.py`` and its
regular reward function. Use it only for OPD runs:

   --custom-rm-path on_policy_distillation.reward_func
   --custom-reward-post-process-path on_policy_distillation.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate

The teacher is queried in SGLang logprob-only mode over the full student
sequence. ``post_process_rewards`` trims teacher log-probs to the response
span and stores them on each sample for Slime's OPD KL path.

This base variant is pure OPD: scalar task rewards returned to GRPO are 0.0.
Use ``on_policy_distillation_v2.py`` for Search-R1 task reward + OPD.
"""

from __future__ import annotations

import asyncio

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import sys

import aiohttp
import torch

from qa_em_format import compute_score_em
from slime.utils.types import Sample


_session_lock = asyncio.Lock()
_session_by_loop: dict[int, aiohttp.ClientSession] = {}
_semaphore_by_loop: dict[int, asyncio.Semaphore] = {}
_TEACHER_MAX_INFLIGHT = 64
_TEACHER_REQUEST_TIMEOUT = 600
_TEACHER_TOTAL_BUDGET = 900
_TEACHER_MAX_RETRIES = 2
_SEARCH_R1_FORMAT_SCORE = 0.2


def _prompt_to_text(prompt) -> str:
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


def _ground_truth_from_label(label):
    if isinstance(label, dict) and "ground_truth" in label:
        return label["ground_truth"]
    return label if label is not None else {"target": []}


def _get_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = id(loop)
    sem = _semaphore_by_loop.get(key)
    if sem is None:
        sem = asyncio.Semaphore(_TEACHER_MAX_INFLIGHT)
        _semaphore_by_loop[key] = sem
    return sem


async def _get_session() -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    key = id(loop)
    sess = _session_by_loop.get(key)
    if sess is not None and not sess.closed:
        return sess
    async with _session_lock:
        sess = _session_by_loop.get(key)
        if sess is not None and not sess.closed:
            return sess
        timeout = aiohttp.ClientTimeout(
            total=_TEACHER_REQUEST_TIMEOUT,
            connect=60,
            sock_connect=60,
            sock_read=300,
        )
        connector = aiohttp.TCPConnector(
            limit=128,
            limit_per_host=128,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True,
        )
        sess = aiohttp.ClientSession(
            trust_env=False,
            timeout=timeout,
            connector=connector,
        )
        _session_by_loop[key] = sess
        return sess


async def reward_func(args, sample: Sample, **kwargs):
    """Query teacher log-probs and compute Search-R1 EM score for monitoring."""
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

    last_err: Exception | None = None
    teacher_response = None
    sem = _get_semaphore()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TEACHER_TOTAL_BUDGET
    for attempt in range(_TEACHER_MAX_RETRIES):
        if deadline - loop.time() <= 0:
            break
        try:
            session = await _get_session()
            async with sem:
                async with session.post(args.rm_url, json=payload) as resp:
                    resp.raise_for_status()
                    teacher_response = await resp.json()
            break
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = e
            await asyncio.sleep(min(2 ** attempt, 8))
        except RuntimeError as e:
            msg = str(e)
            if "File descriptor" not in msg and "transport" not in msg:
                raise
            last_err = e
            await asyncio.sleep(0.05 * (attempt + 1))

    if teacher_response is None:
        print(
            f"[search-opd] WARN: teacher /generate failed after "
            f"{_TEACHER_MAX_RETRIES} retries (url={args.rm_url}, "
            f"tokens={len(sample.tokens)}): {last_err!r}",
            file=sys.stderr,
            flush=True,
        )
        sample._opd_teacher_response = None  # type: ignore[attr-defined]
        sample._opd_task_score = 0.0  # type: ignore[attr-defined]
        return 0.0

    solution_str = _prompt_to_text(sample.prompt) + sample.response
    task_score = float(
        compute_score_em(
            solution_str=solution_str,
            ground_truth=_ground_truth_from_label(sample.label),
            format_score=_SEARCH_R1_FORMAT_SCORE,
        )
    )

    sample._opd_teacher_response = teacher_response  # type: ignore[attr-defined]
    sample._opd_task_score = task_score  # type: ignore[attr-defined]
    return task_score


def _extract_teacher_log_probs(samples: list[Sample]) -> list[torch.Tensor]:
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
            f"[search-opd] WARN: {n_failed}/{len(samples)} samples had no teacher "
            f"logprobs; substituted zeros.",
            file=sys.stderr,
            flush=True,
        )
    return teacher_log_probs


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Attach teacher log-probs and return zero scalar rewards for pure OPD."""
    for sample, t_log_probs in zip(samples, _extract_teacher_log_probs(samples), strict=False):
        sample.teacher_log_probs = t_log_probs

    task_scores = [float(getattr(sample, "_opd_task_score", 0.0) or 0.0) for sample in samples]
    try:
        import wandb

        if wandb.run is not None and task_scores:
            wandb.log({"rollout/student_task_score": sum(task_scores) / len(task_scores)})
    except ImportError:
        pass

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
