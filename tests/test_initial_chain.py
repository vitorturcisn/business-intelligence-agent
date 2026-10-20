
"""Testes locais do schema e do parser da chain inicial."""

import pytest
from pydantic import ValidationError
from langchain_core.output_parsers import PydanticOutputParser

from business_agent.chains.initial_chain import RequestAnalysis


def test_parser_converts_valid_json():
    parser = PydanticOutputParser(
        pydantic_object=RequestAnalysis
    )

    raw_json = """
    {
        "intent": "sales_analysis",
        "metric": "faturamento",
        "period": "2026-09",
        "requires_tools": true,
        "missing_information": []
    }
    """

    result = parser.parse(raw_json)

    assert isinstance(result, RequestAnalysis)
    assert result.intent == "sales_analysis"
    assert result.metric == "faturamento"
    assert result.period == "2026-09"
    assert result.requires_tools is True


def test_schema_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        RequestAnalysis(
            intent="unknown_intent",
            requires_tools=False,
        )


def test_schema_accepts_missing_optional_fields():
    result = RequestAnalysis(
        intent="other",
        requires_tools=False,
    )

    assert result.metric is None
    assert result.period is None
    assert result.missing_information == []
