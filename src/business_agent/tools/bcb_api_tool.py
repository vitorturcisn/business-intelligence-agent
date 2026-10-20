
"""Ferramenta de consulta à meta da Selic na API do Banco Central."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

import requests
from langchain_core.tools import tool


API_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados"
MAX_PERIOD_DAYS = 3650


def _parse_date(value: str, field_name: str) -> date:
    """Valida uma data informada no formato ISO YYYY-MM-DD."""

    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", value
    ):
        raise ValueError(
            f"{field_name} deve estar no formato YYYY-MM-DD."
        )

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} não é uma data válida: {value}."
        ) from exc


def fetch_selic_observations(
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Consulta observações da meta Selic em um período.

    Se nenhuma data for informada, consulta os últimos 30 dias.
    Se uma data for informada, a outra também será obrigatória.
    """

    today = date.today()

    if start_date is None and end_date is None:
        end = today
        start = end - timedelta(days=30)
    elif start_date is not None and end_date is not None:
        start = _parse_date(start_date, "start_date")
        end = _parse_date(end_date, "end_date")
    else:
        raise ValueError(
            "Informe start_date e end_date juntos, ou deixe ambos vazios."
        )

    if start > end:
        raise ValueError("A data inicial não pode ser posterior à final.")

    if (end - start).days > MAX_PERIOD_DAYS:
        raise ValueError(
            "O intervalo não pode ultrapassar 3650 dias."
        )

    # A API recebe as datas no formato DD/MM/AAAA.
    params = {
        "formato": "json",
        "dataInicial": start.strftime("%d/%m/%Y"),
        "dataFinal": end.strftime("%d/%m/%Y"),
    }

    try:
        response = requests.get(
            API_URL,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Falha ao consultar a API do Banco Central: {exc}"
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            "A API do Banco Central devolveu uma resposta JSON inválida."
        ) from exc

    if not isinstance(payload, list):
        raise RuntimeError(
            "Formato inesperado na resposta da API do Banco Central."
        )

    observations = []

    for item in payload:
        if not isinstance(item, dict):
            raise RuntimeError("Observação inválida recebida da API.")

        try:
            observation_date = _parse_bcb_date(item["data"])
            value = float(str(item["valor"]).replace(",", "."))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Não foi possível interpretar uma observação da Selic."
            ) from exc

        observations.append(
            {
                "date": observation_date.isoformat(),
                "value": value,
                "unit": "% a.a.",
            }
        )

    observations.sort(key=lambda item: item["date"])

    return {
        "series_code": 432,
        "description": "Meta da taxa Selic definida pelo Copom",
        "period": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "unit": "percentual ao ano",
        "observations_count": len(observations),
        "latest_observation": (
            observations[-1] if observations else None
        ),
        "observations": observations,
        "source": API_URL,
    }


def _parse_bcb_date(value: str) -> date:
    """Converte uma data do Banco Central, DD/MM/AAAA, para date."""

    if not isinstance(value, str):
        raise ValueError("A data da observação deve ser texto.")

    try:
        return date.strptime(value, "%d/%m/%Y")
    except AttributeError:
        # Compatibilidade com versões Python sem date.strptime.
        from datetime import datetime

        return datetime.strptime(value, "%d/%m/%Y").date()
    except ValueError as exc:
        raise ValueError(
            f"Data inválida retornada pela API: {value}."
        ) from exc


@tool("get_selic_target")
def get_selic_target(
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Consulta a meta Selic no Banco Central do Brasil.

    Use sem parâmetros para consultar os últimos 30 dias e encontrar
    a observação mais recente disponível.

    Para consultar um período específico, informe start_date e end_date
    no formato YYYY-MM-DD. O período máximo permitido é de 3650 dias.

    Retorna as observações, datas, unidade percentual ao ano e a fonte.
    Esta ferramenta consulta a API real e necessita de acesso à internet.
    """

    result = fetch_selic_observations(
        start_date=start_date,
        end_date=end_date,
    )

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )
