# Adapted from search-r1/generate_with_search.py.
# This is the eval-only variant: we strip the slime-specific `generate()` /
# `reward_func()` entry points so the file can be imported by `eval.py`
# without needing the slime training stack.
import asyncio
import re

# Configuration for Search-R1 eval. Override SEARCH_R1_CONFIGS["search_backend"],
# SEARCH_R1_CONFIGS["local"]["search_url"], etc. before calling search().
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 4,
    "topk": 3,
    "search_concurrency": 256,
    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google"
    # ============== Local Search Configuration ==============
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",
        "proxy": None,
    },
    # ============== Google Search Configuration ==============
    "google": {
        "api_key": "your_api_key_here",
        "snippet_only": True,
        "proxy": None,
    },
    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])


def _passages2string(retrieval_result):
    """Convert retrieval results to a formatted reference string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


async def search(query: str) -> str:
    """Perform search using either local search engine or Google search."""
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
        raise ValueError(
            f"Unknown search backend: {backend}. Must be either 'local' or 'google'."
        )

    return _passages2string(result)


def postprocess_responses(resp: str) -> str:
    """Truncate the assistant response at the first closing </search> or </answer> tag."""
    if "</search>" in resp:
        return resp.split("</search>")[0] + "</search>"
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    return resp


def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    """Return (action, content) where action is 'search'|'answer'|None."""
    pattern = r"<(search|answer)>(.*?)</\1>"
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    return None, ""


async def execute_predictions(prediction: str) -> tuple[str, bool]:
    """Run the action implied by `prediction`. Return (next_observation, done)."""
    action, content = postprocess_predictions(prediction)

    if action == "search":
        async with SEMAPHORE:
            search_results = await search(content)
        next_obs = f"\n\n<information>{search_results.strip()}</information>\n\n"
        return next_obs, False
    if action == "answer":
        return "", True
    next_obs = (
        "\nMy previous action is invalid. "
        "If I want to search, I should put the query between <search> and </search>. "
        "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
        "Let me try again.\n"
    )
    return next_obs, False
