
"""Testes da integração com a API do Banco Central."""

from unittest.mock import Mock, patch

import pytest
import requests

from business_agent.tools.bcb_api_tool import (
    fetch_selic_observations,
    get_selic_target,
)


def test_parses_api_response_and_sorts_observations():
    fake_response = Mock()
    fake_response.json.return_value = [
        {"data": "20/10/2026", "valor": "15.00"},
        {"data": "19/10/2026", "valor": "15.00"},
    ]

    with patch(
        "business_agent.tools.bcb_api_tool.requests.get",
        return_value=fake_response,
    ) as mocked_get:
        result = fetch_selic_observations(
            "2026-10-01",
            "2026-10-20",
        )

    assert result["series_code"] == 432
    assert result["observations_count"] == 2
    assert result["observations"][0]["date"] == "2026-10-19"
    assert result["latest_observation"]["date"] == "2026-10-20"
    assert result["latest_observation"]["value"] == 15.0

    params = mocked_get.call_args.kwargs["params"]
    assert params["dataInicial"] == "01/10/2026"
    assert params["dataFinal"] == "20/10/2026"
    fake_response.raise_for_status.assert_called_once()


def test_invalid_date_is_rejected_without_http_call():
    with patch(
        "business_agent.tools.bcb_api_tool.requests.get"
    ) as mocked_get:
        with pytest.raises(ValueError):
            fetch_selic_observations(
                "2026-13-01",
                "2026-10-20",
            )

    mocked_get.assert_not_called()


def test_both_dates_must_be_provided_together():
    with pytest.raises(ValueError, match="juntos"):
        fetch_selic_observations(
            start_date="2026-10-01",
        )


def test_start_date_cannot_be_after_end_date():
    with pytest.raises(ValueError, match="posterior"):
        fetch_selic_observations(
            "2026-10-20",
            "2026-10-01",
        )


def test_http_error_is_reported():
    fake_response = Mock()
    fake_response.raise_for_status.side_effect = (
        requests.HTTPError("HTTP 500")
    )

    with patch(
        "business_agent.tools.bcb_api_tool.requests.get",
        return_value=fake_response,
    ):
        with pytest.raises(RuntimeError, match="Falha ao consultar"):
            fetch_selic_observations(
                "2026-10-01",
                "2026-10-20",
            )


def test_langchain_tool_is_registered():
    assert get_selic_target.name == "get_selic_target"
    assert "Banco Central" in get_selic_target.description
