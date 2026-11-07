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
        config={"configurable": {"thread_id": thread_id}},
    )


def test_agent_registers_all_business_tools():
    names = {tool.name for tool in get_business_tools()}

    assert names == {
        "sales_monthly_summary",
        "sales_category_ranking",
        "business_knowledge_search",
        "get_selic_target",
        "save_business_memory",
        "search_business_memory",
        "get_business_memory",
    }


def test_agent_executes_real_sales_tool():
    agent = create_test_agent()

    result = invoke(
        agent,
        "Apresente as vendas de setembro de 2026.",
        "test-sales-agent",
    )

    tool_messages = [
        message for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].name == "sales_monthly_summary"
    assert "2026-09" in tool_messages[0].content
    assert "25250" in tool_messages[0].content


def test_agent_preserves_history_by_thread_id():
    agent = create_test_agent()
    thread_id = "test-memory-agent"

    first = invoke(
        agent,
        "Consulte as vendas de setembro de 2026.",
        thread_id,
    )

    second = invoke(
        agent,
        "Agora continue nossa análise.",
        thread_id,
    )

    users = [
        message for message in second["messages"]
        if isinstance(message, HumanMessage)
    ]

    assert len(users) == 2
    assert len(second["messages"]) > len(first["messages"])


def test_agent_reuses_previous_tool_result_without_repeating_sql():
    agent = create_test_agent()
    thread_id = "test-reuse-memory-agent"

    first = invoke(
        agent,
        "Consulte as vendas de setembro de 2026.",
        thread_id,
    )

    previous_count = len(first["messages"])

    second = invoke(
        agent,
        "Agora explique o resultado anterior.",
        thread_id,
    )

    new_messages = second["messages"][previous_count:]

    new_tools = [
        message for message in new_messages
        if isinstance(message, ToolMessage)
    ]

    new_calls = [
        call
        for message in new_messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]

    assert not new_tools
    assert not new_calls
    assert "25250" in second["messages"][-1].content


def test_agent_sessions_do_not_share_tool_history():
    agent = create_test_agent()

    invoke(
        agent,
        "Consulte as vendas de setembro de 2026.",
        "session-A",
    )

    result = invoke(
        agent,
        "Continue a análise anterior.",
        "session-B",
    )

    tools = [
        message for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tools) == 1
    assert tools[0].name == "sales_monthly_summary"
