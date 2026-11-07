from langchain_core.messages import AIMessage, ToolMessage

from langgraph.checkpoint.memory import InMemorySaver

from business_agent.agents.orchestrator import build_business_agent
from business_agent.agents.mock_model import FakeSalesToolCallingModel


def main():
    model = FakeSalesToolCallingModel(responses=[])
    checkpointer = InMemorySaver()

    agent = build_business_agent(
        model=model,
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {
            "thread_id": "demonstracao-business-intelligence-001"
        }
    }

    questions = [
        "Apresente o resumo de vendas de setembro de 2026.",
        "Agora considere também o contexto da nossa conversa anterior.",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print("USUÁRIO:", question)
        print("=" * 80)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )

        for message in result["messages"]:
            if isinstance(message, AIMessage) and message.tool_calls:
                for call in message.tool_calls:
                    print(f"\nCHAMADA DE FERRAMENTA: {call['name']}")
                    print(f"ARGUMENTOS: {call['args']}")

            elif isinstance(message, ToolMessage):
                print(f"\nRESULTADO DA FERRAMENTA:\n{message.content}")

        print("\nRESPOSTA FINAL:")
        print(result["messages"][-1].content)

        print(
            "\nMENSAGENS NO HISTÓRICO:",
            len(result["messages"]),
        )


if __name__ == "__main__":
    main()
