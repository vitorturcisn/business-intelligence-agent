from business_agent.agents.legacy_openai_tools import (
    build_legacy_openai_tools_executor,
)


def main():
    executor = build_legacy_openai_tools_executor(
        verbose=False,
    )

    question = "Apresente as vendas de setembro de 2026."

    print("=" * 75)
    print("BUSINESS INTELLIGENCE AGENT")
    print("Demonstração: create_openai_tools_agent + AgentExecutor")
    print("=" * 75)

    print("\nPergunta:")
    print(question)

    result = executor.invoke(
        {"input": question}
    )

    print("\nRESPOSTA FINAL")
    print("-" * 75)
    print(result["output"])

    print("\nTRAJETÓRIA DE EXECUÇÃO")
    print("-" * 75)

    for index, (action, observation) in enumerate(
        result.get("intermediate_steps", []),
        start=1,
    ):
        print(f"\nEtapa {index}")
        print("Ferramenta:", action.tool)
        print("Argumentos:", action.tool_input)
        print("Resultado:", observation)

    print("\n" + "=" * 75)
    print("Demonstração concluída.")
    print("=" * 75)


if __name__ == "__main__":
    main()
