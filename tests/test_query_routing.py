from business_agent.agents.query_routing import is_latest_selic_query


def test_latest_selic_query_detects_portuguese_latest_request():
    assert is_latest_selic_query(
        "Qual é a meta Selic mais recente?"
    )


def test_latest_selic_query_detects_sgs_identifier_and_latest_observation():
    assert is_latest_selic_query(
        "Informe a última observação da série SGS 432."
    )


def test_latest_selic_query_detects_explicit_tool_name_and_current_value():
    assert is_latest_selic_query(
        "Consulte get_selic_target e informe o valor atual."
    )


def test_historical_selic_period_is_not_routed_as_latest():
    assert not is_latest_selic_query(
        "Consulte a taxa Selic entre 2026-09-01 e 2026-09-30."
    )


def test_unrelated_latest_question_is_not_routed_to_selic_api():
    assert not is_latest_selic_query(
        "Qual foi a receita líquida mais recente?"
    )


def test_explicit_historical_period_is_not_routed_as_latest():
    assert not is_latest_selic_query(
        "Qual foi a observação mais recente da Selic em setembro de 2026?"
    )
