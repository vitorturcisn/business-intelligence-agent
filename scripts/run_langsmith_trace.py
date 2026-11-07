from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client, tracing_context

from business_agent.agents.orchestrator import build_business_agent
from business_agent.agents.mock_model import FakeSalesToolCallingModel
from business_agent.observability.langsmith_config import (
    get_langsmith_status,
)


def main():
    load_dotenv()

    status = get_langsmith_status()

    agent = build_business_agent(
        model=FakeSalesToolCallingModel(responses=[]),
        checkpointer=InMemorySaver(),
    )

    print("Executando agente...")
    print("Projeto:", status.project)
    print("Tracing ativo:", status.tracing_enabled)

    with tracing_context(
        enabled=status.tracing_enabled,
        project_name=status.project,
        tags=[
            "business-intelligence-agent",
            "sql",
            "mock-model",
        ],
        metadata={
            "evaluation_mode": "deterministic_mock",
            "environment": "development",
        },
    ):
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Apresente as vendas de setembro de 2026."
                        ),
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "langsmith-tracing-demo",
                }
            },
        )

    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            print("\nFerramenta executada:", message.name)
            print("Resultado:", message.content)

    print("\nResposta final:")
    print(result["messages"][-1].content)

    if status.tracing_enabled:
        Client().flush()
        print(
            f"\nTrace enviado para o projeto: {status.project}"
        )
    else:
        print(
            "\nExecução apenas local. Configure a API key e "
            "LANGSMITH_TRACING=true para enviar o trace."
        )


if __name__ == "__main__":
    main()
