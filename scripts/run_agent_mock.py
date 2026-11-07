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

    previous_message_count = 0

    for question in questions:
        print("\n" + "=" * 80)
        print("USUÁRIO:", question)
        print("=" * 80)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )

        new_messages = result["messages"][previous_message_count:]

        for message in new_messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                for call in message.tool_calls:
                    print(f"\nCHAMADA DE FERRAMENTA: {call['name']}")
                    print(f"ARGUMENTOS: {call['args']}")

            elif isinstance(message, ToolMessage):
                print(f"\nRESULTADO DA FERRAMENTA:\n{message.content}")

        print("\nRESPOSTA FINAL:")
        print(result["messages"][-1].content)

        previous_message_count = len(result["messages"])

        print("\nTOTAL DE MENSAGENS NO HISTÓRICO:", previous_message_count)


if __name__ == "__main__":
    main()
