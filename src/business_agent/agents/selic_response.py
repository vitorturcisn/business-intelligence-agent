from __future__ import annotations

import json
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any


def _decode_observation(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("A ferramenta retornou JSON inválido.") from exc
    else:
        raise TypeError("A observação da ferramenta deve ser JSON ou dict.")

    if not isinstance(payload, dict):
        raise ValueError("O retorno da ferramenta não é um objeto JSON.")
    latest = payload.get("latest_observation")
    if not isinstance(latest, dict):
        raise ValueError("O retorno não contém latest_observation válido.")
    return latest


def build_latest_selic_answer(intermediate_steps: list[tuple[Any, Any]]) -> str:
    """Build a factual answer from exactly one real get_selic_target tool call."""
    observations = []
    for step in intermediate_steps or []:
        if not isinstance(step, (tuple, list)) or len(step) != 2:
            continue
        action, observation = step
        if getattr(action, "tool", None) == "get_selic_target":
            observations.append(observation)

    if len(observations) != 1:
        raise ValueError(
            "Era esperada exatamente uma chamada a get_selic_target; "
            f"foram encontradas {len(observations)}."
        )

    latest = _decode_observation(observations[0])
    raw_date = latest.get("date")
    raw_value = latest.get("value")

    if not isinstance(raw_date, str):
        raise ValueError("A data da última observação está ausente ou inválida.")
    try:
        observation_date = date.fromisoformat(raw_date)
    except ValueError as exc:
        raise ValueError("A data da última observação não está em ISO YYYY-MM-DD.") from exc

    if isinstance(raw_value, bool) or raw_value is None:
        raise ValueError("O valor da última observação está ausente ou inválido.")
    try:
        value = Decimal(str(raw_value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("O valor da última observação não é numérico.") from exc
    if not value.is_finite():
        raise ValueError("O valor da última observação não é finito.")

    formatted_value = format(value, ".2f").replace(".", ",")
    return (
        f"A meta Selic mais recente foi de {formatted_value}% a.a., "
        f"conforme a última observação da série SGS 432 em "
        f"{observation_date.isoformat()}."
    )
