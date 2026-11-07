from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from business_agent.agents.orchestrator import (
    build_business_agent,
    get_business_tools,
)
from business_agent.agents.mock_model import FakeSalesToolCallingModel


def create_test_agent():
    return build_business_agent(
        model=FakeSalesToolCallingModel(responses=[]),
        checkpointer=InMemorySaver(),
    )


def invoke(agent, question, thread_id):
    return agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )


def test_agent_registers_all_business_tools():
    tools = get_business_tools()

    names = {tool.name for tool in tools}

    assert names == {
        "sales_monthly_summary",
        "sales_category_ranking",
        "business_knowledge_search",
        "get_selic_target",
    }


def test_agent_executes_real_sales_tool():
    agent = create_test_agent()

    result = invoke(
        agent,
        "Apresente as vendas de setembro de 2026.",
        "test-sales-agent",
    )

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].name == "sales_monthly_summary"
    assert "2026-09" in tool_messages[0].content
    assert "25250" in tool_messages[0].content
    assert result["messages"][-1].content


def test_agent_preserves_history_by_thread_id():
    agent = create_test_agent()
    thread_id = "test-memory-agent"

    first_result = invoke(
        agent,
        "Consulte as vendas de setembro de 2026.",
        thread_id,
    )

    second_result = invoke(
        agent,
        "Agora continue nossa análise.",
        thread_id,
    )

    user_messages = [
        message
        for message in second_result["messages"]
        if isinstance(message, HumanMessage)
    ]

    assert len(user_messages) == 2

    assert user_messages[0].content == (
        "Consulte as vendas de setembro de 2026."
    )

    assert user_messages[1].content == (
        "Agora continue nossa análise."
    )

    assert len(second_result["messages"]) > len(first_result["messages"])


def test_agent_reuses_previous_tool_result_without_repeating_sql():
    agent = create_test_agent()
    thread_id = "test-reuse-memory-agent"

    first_result = invoke(
        agent,
        "Consulte as vendas de setembro de 2026.",
        thread_id,
    )

    previous_message_count = len(first_result["messages"])

    second_result = invoke(
        agent,
        "Agora explique o resultado anterior.",
        thread_id,
    )

    new_messages = second_result["messages"][previous_message_count:]

    new_tool_messages = [
        message
        for message in new_messages
        if isinstance(message, ToolMessage)
    ]

    new_tool_calls = [
        call
        for message in new_messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]

    assert len(new_tool_messages) == 0
    assert len(new_tool_calls) == 0

    assert "25250" in second_result["messages"][-1].content
    assert "consulta anterior" in second_result["messages"][-1].content.lower()


def test_agent_sessions_do_not_share_tool_history():
    agent = create_test_agent()

    invoke(
        agent,
        "Consulte as vendas de setembro de 2026.",
        "session-A",
    )

    result_b = invoke(
        agent,
        "Continue a análise anterior.",
        "session-B",
    )

    tool_messages = [
        message
        for message in result_b["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].name == "sales_monthly_summary"
