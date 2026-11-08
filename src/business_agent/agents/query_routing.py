from __future__ import annotations

import re
import unicodedata


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", without_accents).strip()


def is_latest_selic_query(question: str) -> bool:
    """Detect requests for the latest/current Selic observation.

    Explicit historical-period questions are intentionally excluded so that
    they can continue through the ordinary AgentExecutor flow.
    """
    if not isinstance(question, str) or not question.strip():
        return False

    normalized = _normalize(question)
    explicit_historical_period = bool(
        re.search(r"\b(?:19|20)\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?\b", normalized)
        or re.search(
            r"\b(?:janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|"
            r"setembro|outubro|novembro|dezembro)\s+(?:de\s+)?(?:19|20)\d{2}\b",
            normalized,
        )
    )
    if explicit_historical_period:
        return False

    refers_to_selic = bool(
        re.search(r"\bselic\b", normalized)
        or re.search(r"\bsgs\s*432\b", normalized)
        or "get_selic_target" in normalized
    )
    if not refers_to_selic:
        return False

    latest_markers = (
        "mais recente",
        "ultima observacao",
        "ultima taxa",
        "valor atual",
        "taxa atual",
        "selic atual",
        "hoje",
        "agora",
    )
    return any(marker in normalized for marker in latest_markers)
