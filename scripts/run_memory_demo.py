import json

from business_agent.memory.long_term import LongTermMemoryStore
from business_agent.memory.tools import (
    save_business_memory,
    search_business_memory,
    get_business_memory,
)


def main():
    memories = [
        {
            "kind": "semantic",
            "memory_key": "kpi_receita_liquida",
            "content": (
                "Receita líquida é receita bruta menos descontos. "
                "A definição está documentada no glossário de KPIs."
            ),
            "tags": ["receita", "kpi", "vendas"],
        },
        {
            "kind": "episodic",
            "memory_key": "analise_vendas_2026_09",
            "content": (
                "Análise do dataset sintético de setembro de 2026: "
                "6 pedidos, 40 unidades, receita bruta de R$ 26.650,00, "
                "descontos de R$ 1.400,00 e receita líquida de "
                "R$ 25.250,00. Fonte: sales_monthly_summary."
            ),
            "tags": ["vendas", "setembro", "2026", "sql"],
        },
        {
            "kind": "procedural",
            "memory_key": "processo_analise_mensal",
            "content": (
                "Para uma análise mensal, consultar o SQL para obter "
                "os valores, utilizar o RAG para conferir definições "
                "dos indicadores e apresentar as fontes utilizadas."
            ),
            "tags": ["processo", "sql", "rag", "analise"],
        },
    ]

    print("SALVANDO MEMÓRIAS...")

    for memory in memories:
        result = save_business_memory.invoke(memory)
        print(json.dumps(json.loads(result), indent=2, ensure_ascii=False))

    print("\nPESQUISANDO MEMÓRIA SEMÂNTICA...")

    semantic_results = search_business_memory.invoke({
        "query": "receita líquida KPI",
        "kind": "semantic",
        "limit": 5,
    })

    print(json.dumps(
        json.loads(semantic_results),
        indent=2,
        ensure_ascii=False,
    ))

    print("\nRECUPERANDO MEMÓRIA EPISÓDICA...")

    episodic_result = get_business_memory.invoke({
        "kind": "episodic",
        "memory_key": "analise_vendas_2026_09",
    })

    print(json.dumps(
        json.loads(episodic_result),
        indent=2,
        ensure_ascii=False,
    ))

    print("\nVALIDANDO PERSISTÊNCIA EM NOVA INSTÂNCIA...")

    reopened = LongTermMemoryStore()

    persisted = reopened.get_memory(
        "procedural",
        "processo_analise_mensal",
    )

    print("Memória procedural recuperada:", persisted is not None)

    if persisted is None:
        raise RuntimeError("A memória não persistiu no SQLite.")


if __name__ == "__main__":
    main()
