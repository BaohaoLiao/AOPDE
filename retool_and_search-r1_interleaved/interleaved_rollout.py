from __future__ import annotations

from typing import Any

from slime.rollout.sglang_rollout import generate_rollout as _generate_rollout


def generate_rollout(args, rollout_id: int, data_source: Any, evaluation: bool = False):
    if not evaluation and hasattr(data_source, "set_rollout_id"):
        data_source.set_rollout_id(rollout_id)
    return _generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
