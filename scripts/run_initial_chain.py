
"""Executa a chain inicial em diferentes modos."""

import argparse

from business_agent.chains.initial_chain import build_chains


def main():
    parser = argparse.ArgumentParser(
        description="Executa o MVP do Business Intelligence Agent."
    )
    parser.add_argument(
        "--mode",
        choices=["json", "imperative", "simple", "batch", "stream"],
        default="json",
        help="Modo de execução da chain.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Pergunta para enviar ao modelo.",
    )
    args = parser.parse_args()

    question = args.question or input("Digite sua pergunta: ").strip()
    if not question:
        raise SystemExit("A pergunta não pode estar vazia.")

    chains = build_chains()

    if args.mode == "json":
        result = chains["json_chain"].invoke({"question": question})
        print("\nResultado estruturado via LCEL:")
        print(result.model_dump_json(indent=2))

    elif args.mode == "imperative":
        result = chains["imperative_json_chain"].invoke(
            {"question": question}
        )
        print("\nResultado estruturado via @chain:")
        print(result.model_dump_json(indent=2))

    elif args.mode == "simple":
        result = chains["simple_chain"].invoke({"question": question})
        print("\nResposta da chain simples:")
        print(result)

    elif args.mode == "batch":
        questions = [
            question,
            "Explique como se calcula a receita líquida de vendas.",
        ]
        results = chains["json_chain"].batch(
            [{"question": item} for item in questions]
        )
        print("\nResultados do batch:")
        for index, result in enumerate(results, start=1):
            print(f"\nCaso {index}:")
            print(result.model_dump_json(indent=2))

    elif args.mode == "stream":
        print("\nResposta em streaming:")
        for chunk in chains["simple_chain"].stream(
            {"question": question}
        ):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
