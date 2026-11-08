import json
from types import SimpleNamespace

import pytest

from business_agent.agents.selic_response import build_latest_selic_answer


def _step(observation):
    return (SimpleNamespace(tool="get_selic_target"), observation)


def test_build_latest_selic_answer_uses_latest_observation_from_json():
    payload = {
        "series_code": 432,
        "latest_observation": {
            "date": "2026-11-08",
            "value": 13.75,
            "unit": "% a.a.",
        },
    }
    answer = build_latest_selic_answer([_step(json.dumps(payload))])
    assert answer == (
        "A meta Selic mais recente foi de 13,75% a.a., conforme a última "
        "observação da série SGS 432 em 2026-11-08."
    )


def test_build_latest_selic_answer_accepts_dict_observation():
    payload = {
        "latest_observation": {
            "date": "2026-11-08",
            "value": 14.0,
        }
    }
    answer = build_latest_selic_answer([_step(payload)])
    assert "14,00% a.a." in answer


def test_build_latest_selic_answer_requires_exactly_one_tool_call():
    with pytest.raises(ValueError, match="exatamente uma chamada"):
        build_latest_selic_answer([])

    step = _step({"latest_observation": {"date": "2026-11-08", "value": 13.75}})
    with pytest.raises(ValueError, match="exatamente uma chamada"):
        build_latest_selic_answer([step, step])


def test_build_latest_selic_answer_rejects_missing_latest_observation():
    with pytest.raises(ValueError, match="latest_observation"):
        build_latest_selic_answer([_step({"observations": []})])


def test_build_latest_selic_answer_rejects_invalid_date_or_value():
    with pytest.raises(ValueError, match="data"):
        build_latest_selic_answer([_step({"latest_observation": {"date": "08/11/2026", "value": 13.75}})])
    with pytest.raises(ValueError, match="numérico"):
        build_latest_selic_answer([_step({"latest_observation": {"date": "2026-11-08", "value": "n/a"}})])
