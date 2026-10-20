
"""Executa a demonstração ReAct e imprime a trajetória das ferramentas."""

import json

from business_agent.agents.legacy_react import build_react_demo


def main():
    print("Iniciando demonstração ReAct...")

    executor = build_react_demo()

    result = executor.invoke(
        {
            "input": (
                "Consulte o faturamento de setembro de 2026 "
                "e obtenha a observação recente da meta da Selic."
            )
        }
    )

    print("\n=== RESPOSTA FINAL ===")
    print(result["output"])

    steps = result.get("intermediate_steps", [])
    print(f"\n=== TRAJETÓRIA: {len(steps)} AÇÃO(ÕES) ===")

    for index, (action, observation) in enumerate(steps, start=1):
        print(f"\nEtapa {index}")
        print(f"Ferramenta: {action.tool}")
        print(f"Entrada: {action.tool_input}")

        if action.tool == "get_selic_recent":
            try:
                data = json.loads(observation)
                print(
                    "Quantidade de observações:",
                    data.get("observations_count"),
                )
                print(
                    "Última observação:",
                    data.get("latest_observation"),
                )
                print("Fonte:", data.get("source"))
            except (TypeError, ValueError):
                print("Resultado:", observation)
        else:
            print("Resultado:")
            print(observation)

    if not steps:
        raise RuntimeError(
            "O agente terminou sem executar ferramentas. "
            "A demonstração não foi concluída."
        )


if __name__ == "__main__":
    main()
