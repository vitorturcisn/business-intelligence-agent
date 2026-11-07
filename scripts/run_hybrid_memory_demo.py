from pathlib import Path

from business_agent.memory.long_term import LongTermMemoryStore
from business_agent.memory.hybrid_retrieval import HybridMemoryRetriever


def main():
    database = Path("data/sample/hybrid_memory_demo.db")
    store = LongTermMemoryStore(database)

    store.save_memory(
        kind="semantic",
        memory_key="demo_receita_liquida",
        content=(
            "Receita líquida corresponde ao faturamento bruto "
            "após a dedução dos descontos comerciais e devoluções."
        ),
        tags=["kpi", "vendas", "receita"],
    )

    store.save_memory(
        kind="episodic",
        memory_key="demo_analise_vendas",
        content=(
            "Uma análise comercial identificou vendas de "
            "setembro de 2026 com seis pedidos, quarenta unidades "
            "e receita líquida de R$ 25.250."
        ),
        tags=["vendas", "setembro", "2026"],
    )

    question = (
        "Quanto sobra do valor vendido depois de retirar "
        "os abatimentos comerciais?"
    )

    retriever = HybridMemoryRetriever(store)

    lexical_results = store.search_memories(
        question,
        limit=5,
    )

    hybrid_results = retriever.search_hybrid(
        question,
        limit=5,
    )

    print("=" * 75)
    print("DEMONSTRAÇÃO DE MEMÓRIA HÍBRIDA")
    print("=" * 75)
    print("\nPergunta:", question)

    print("\nBusca lexical:")
    print("Resultados:", len(lexical_results))

    print("\nBusca híbrida:")
    print("Resultados:", len(hybrid_results))

    for index, item in enumerate(hybrid_results, start=1):
        print(f"\nResultado {index}")
        print("Chave:", item["memory_key"])
        print("Tipo:", item["kind"])
        print("Conteúdo:", item["content"])
        print("Métodos:", item["retrieval_methods"])
        print("Pontuação RRF:", round(item["hybrid_score"], 5))

        if "semantic_distance" in item:
            print(
                "Distância FAISS:",
                round(item["semantic_distance"], 5),
            )

    print("\nBanco SQLite:", database.resolve())
    print("Índice FAISS:", retriever.index_dir.resolve())
    print("\nDemonstração concluída.")


if __name__ == "__main__":
    main()
