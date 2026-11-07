from business_agent.observability.langsmith_config import (
    get_langsmith_status,
)


def test_tracing_is_disabled_without_api_key():
    status = get_langsmith_status({
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "",
        "LANGSMITH_PROJECT": "test",
    })

    assert status.api_key_set is False
    assert status.tracing_enabled is False


def test_tracing_is_enabled_with_key_and_flag():
    status = get_langsmith_status({
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "lsv2_test_key",
        "LANGSMITH_PROJECT": "business-agent-test",
    })

    assert status.api_key_set is True
    assert status.tracing_enabled is True
    assert status.project == "business-agent-test"


def test_api_key_does_not_enable_tracing_by_itself():
    status = get_langsmith_status({
        "LANGSMITH_TRACING": "false",
        "LANGSMITH_API_KEY": "lsv2_test_key",
    })

    assert status.api_key_set is True
    assert status.tracing_enabled is False
