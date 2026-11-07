import os

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class LangSmithStatus:
    api_key_set: bool
    tracing_requested: bool
    tracing_enabled: bool
    project: str


def get_langsmith_status(
    environ: Mapping[str, str] | None = None,
) -> LangSmithStatus:

    env = environ if environ is not None else os.environ

    api_key = env.get("LANGSMITH_API_KEY", "").strip()

    placeholder_values = {
        "",
        "your_api_key_here",
        "your-langsmith-api-key",
        "placeholder",
    }

    api_key_set = api_key.casefold() not in placeholder_values

    tracing_requested = (
        env.get("LANGSMITH_TRACING", "false").strip().lower()
        in {"true", "1", "yes"}
    )

    project = env.get(
        "LANGSMITH_PROJECT",
        "default",
    ).strip() or "default"

    return LangSmithStatus(
        api_key_set=api_key_set,
        tracing_requested=tracing_requested,
        tracing_enabled=api_key_set and tracing_requested,
        project=project,
    )
