from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from business_agent.agents.orchestrator import (
    build_business_agent,
    get_business_tools,
)
from business_agent.agents.mock_model import FakeSalesToolCallingModel


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
    agent = build_business_agent(
        model=FakeSalesToolCallingModel(responses=[]),
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Apresente as vendas de setembro de 2026.",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": "test-sales-agent",
            }
        },
    )

    messages = result["messages"]

    tool_messages = [
        message
        for message in messages
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].name == "sales_monthly_summary"
    assert "2026-09" in tool_messages[0].content
    assert messages[-1].content


def test_agent_preserves_history_by_thread_id():
    agent = build_business_agent(
        model=FakeSalesToolCallingModel(responses=[]),
        checkpointer=InMemorySaver(),
    )

    config = {
        "configurable": {
            "thread_id": "test-memory-agent",
        }
    }

    first_result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Consulte as vendas de setembro de 2026.",
                }
            ]
        },
        config=config,
    )

    second_result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Agora continue nossa análise.",
                }
            ]
        },
        config=config,
    )

    messages = second_result["messages"]

    user_messages = [
        message
        for message in messages
        if isinstance(message, HumanMessage)
    ]

    assert len(user_messages) == 2

    assert any(
        message.content == "Consulte as vendas de setembro de 2026."
        for message in user_messages
    )

    assert len(second_result["messages"]) > len(first_result["messages"])
