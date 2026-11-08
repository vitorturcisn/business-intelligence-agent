"""Avalia a recuperação do RAG usando o índice FAISS já persistido.

Compara três estratégias:
1. original: somente a pergunta original;
2. expanded: somente a pergunta transformada por transform_query;
3. hybrid: pergunta original + transformada, com a mesma união por distância
   utilizada em search_knowledge_base.

As métricas são calculadas no nível da fonte/documento. Quando vários chunks
vierem do mesmo arquivo, a fonte é contada uma única vez, preservando sua
primeira posição. O denominador de Precision@k permanece k.

Este script NÃO reconstrói nem modifica o índice FAISS.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_agent.retrieval.faiss_store import load_faiss_index  # noqa: E402
from business_agent.tools.rag_tool import transform_query  # noqa: E402


K = 4
STRATEGIES = ("original", "expanded", "hybrid")

# Conjunto de avaliação curado a partir dos quatro documentos em
# data/knowledge_base. gold_sources contém os documentos relevantes esperados.
EVALUATION_SET: list[dict[str, Any]] = [
    {
        "id": "RAG-01",
        "question": "Como calcular a receita bruta por item antes dos descontos?",
        "gold_sources": ["01_glossario_kpis.md", "02_dicionario_dados.md"],
    },
    {
        "id": "RAG-02",
        "question": "Como descobrir o faturamento após os descontos?",
        "gold_sources": [
            "01_glossario_kpis.md",
            "02_dicionario_dados.md",
            "03_regras_negocio.md",
        ],
    },
    {
        "id": "RAG-03",
        "question": "Qual é a fórmula da receita líquida para cada item vendido?",
        "gold_sources": ["01_glossario_kpis.md", "02_dicionario_dados.md"],
    },
    {
        "id": "RAG-04",
        "question": "Quais campos da tabela sales são necessários para calcular a receita líquida?",
        "gold_sources": ["02_dicionario_dados.md"],
    },
    {
        "id": "RAG-05",
        "question": "Qual a diferença entre quantidade de produtos vendida e número de pedidos?",
        "gold_sources": [
            "01_glossario_kpis.md",
            "02_dicionario_dados.md",
            "03_regras_negocio.md",
        ],
    },
    {
        "id": "RAG-06",
        "question": "Como contar pedidos sem confundir com as unidades comercializadas?",
        "gold_sources": [
            "01_glossario_kpis.md",
            "02_dicionario_dados.md",
            "03_regras_negocio.md",
        ],
    },
    {
        "id": "RAG-07",
        "question": "Como comparar o faturamento de setembro de 2026 com agosto de 2026?",
        "gold_sources": ["01_glossario_kpis.md", "03_regras_negocio.md"],
    },
    {
        "id": "RAG-08",
        "question": "O que significa variação percentual em uma comparação entre meses?",
        "gold_sources": ["01_glossario_kpis.md", "03_regras_negocio.md"],
    },
    {
        "id": "RAG-09",
        "question": "Como calcular a receita por categoria de produto?",
        "gold_sources": ["01_glossario_kpis.md", "02_dicionario_dados.md"],
    },
    {
        "id": "RAG-10",
        "question": "De onde vêm os dados dos juros básicos da economia brasileira?",
        "gold_sources": ["04_fontes_e_limitacoes.md"],
    },
    {
        "id": "RAG-11",
        "question": "Qual é a série do Banco Central usada para consultar a taxa Selic?",
        "gold_sources": ["04_fontes_e_limitacoes.md"],
    },
    {
        "id": "RAG-12",
        "question": "Quais limitações devo informar sobre os dados de vendas?",
        "gold_sources": ["03_regras_negocio.md", "04_fontes_e_limitacoes.md"],
    },
    {
        "id": "RAG-13",
        "question": "Os dados comerciais são reais ou sintéticos?",
        "gold_sources": ["03_regras_negocio.md", "04_fontes_e_limitacoes.md"],
    },
    {
        "id": "RAG-14",
        "question": "O que acontece se a API de dados do Banco Central estiver indisponível?",
        "gold_sources": ["04_fontes_e_limitacoes.md"],
    },
]


def _chunk_identity(document: Any) -> tuple[Any, Any, str]:
    return (
        document.metadata.get("source"),
        document.metadata.get("start_index"),
        document.page_content,
    )


def _retrieve_chunks(store: Any, query: str, k: int) -> list[tuple[Any, float]]:
    matches = store.similarity_search_with_score(query, k=k)
    return [(document, float(distance)) for document, distance in matches]


def _merge_like_production(
    original_matches: list[tuple[Any, float]],
    expanded_matches: list[tuple[Any, float]],
    k: int,
) -> list[tuple[Any, float]]:
    """Replica a união por identidade de chunk e menor distância do RAG atual."""
    best: dict[tuple[Any, Any, str], tuple[Any, float]] = {}

    for document, distance in [*original_matches, *expanded_matches]:
        identity = _chunk_identity(document)
        previous = best.get(identity)
        if previous is None or distance < previous[1]:
            best[identity] = (document, distance)

    ranked = sorted(best.values(), key=lambda item: item[1])
    return ranked[:k]


def _rank_unique_sources(chunks: list[tuple[Any, float]], k: int) -> list[str]:
    """Colapsa chunks do mesmo arquivo, mantendo a primeira ocorrência."""
    sources: list[str] = []
    seen: set[str] = set()

    for document, _distance in chunks:
        source = str(document.metadata.get("source", "Fonte desconhecida"))
        if source not in seen:
            sources.append(source)
            seen.add(source)
        if len(sources) >= k:
            break

    return sources


def _calculate_metrics(
    ranked_sources: list[str],
    gold_sources: list[str],
    k: int,
) -> dict[str, float]:
    gold = set(gold_sources)
    hits = [source for source in ranked_sources[:k] if source in gold]

    first_relevant_rank = next(
        (rank for rank, source in enumerate(ranked_sources[:k], start=1) if source in gold),
        None,
    )

    return {
        "precision_at_k": len(set(hits)) / k,
        "recall_at_k": len(set(hits)) / len(gold) if gold else 0.0,
        "mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
    }


def evaluate(store: Any, k: int = K) -> dict[str, Any]:
    detail_rows: list[dict[str, Any]] = []

    for item in EVALUATION_SET:
        question = item["question"]
        expanded_query = transform_query(question)

        original_matches = _retrieve_chunks(store, question, k)
        expanded_matches = (
            _retrieve_chunks(store, expanded_query, k)
            if expanded_query != question
            else original_matches
        )

        candidates = {
            "original": original_matches,
            "expanded": expanded_matches,
            "hybrid": _merge_like_production(
                original_matches,
                expanded_matches,
                k,
            ),
        }

        for strategy in STRATEGIES:
            ranked_sources = _rank_unique_sources(candidates[strategy], k)
            metrics = _calculate_metrics(ranked_sources, item["gold_sources"], k)
            detail_rows.append(
                {
                    "question_id": item["id"],
                    "question": question,
                    "expanded_query": expanded_query,
                    "strategy": strategy,
                    "gold_sources": item["gold_sources"],
                    "ranked_sources": ranked_sources,
                    **metrics,
                }
            )

    summary: dict[str, dict[str, float]] = {}
    for strategy in STRATEGIES:
        rows = [row for row in detail_rows if row["strategy"] == strategy]
        summary[strategy] = {
            metric: mean(row[metric] for row in rows)
            for metric in ("precision_at_k", "recall_at_k", "mrr")
        }

    original = summary["original"]
    hybrid = summary["hybrid"]
    comparison = {
        metric: hybrid[metric] - original[metric]
        for metric in ("precision_at_k", "recall_at_k", "mrr")
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_path": "data/indexes/business_knowledge_faiss",
        "index_rebuilt": False,
        "evaluation_level": "document/source; duplicate chunks from the same source are counted once",
        "k": k,
        "number_of_questions": len(EVALUATION_SET),
        "metric_definitions": {
            "precision_at_k": "proporção de fontes relevantes entre as k posições; fontes repetidas são contadas uma vez",
            "recall_at_k": "proporção das fontes relevantes esperadas recuperadas nas k posições",
            "mrr": "recíproco da posição da primeira fonte relevante; zero quando nenhuma fonte relevante aparece",
        },
        "summary": summary,
        "hybrid_minus_original": comparison,
        "details": detail_rows,
    }


def save_results(results: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "rag_retrieval_evaluation.json"
    csv_path = output_dir / "rag_retrieval_evaluation.csv"

    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = [
            "question_id",
            "question",
            "expanded_query",
            "strategy",
            "gold_sources",
            "ranked_sources",
            "precision_at_k",
            "recall_at_k",
            "mrr",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in results["details"]:
            writer.writerow(
                {
                    **row,
                    "gold_sources": " | ".join(row["gold_sources"]),
                    "ranked_sources": " | ".join(row["ranked_sources"]),
                }
            )

    return json_path, csv_path


def main() -> int:
    print("Avaliação de recuperação RAG — índice FAISS existente")
    print(f"Perguntas: {len(EVALUATION_SET)} | k={K}")
    print("Estratégias: original, expanded, hybrid")
    print("O índice NÃO será reconstruído.\n")

    try:
        store = load_faiss_index()
        results = evaluate(store, k=K)
    except FileNotFoundError as exc:
        print(f"ERRO: {exc}")
        print("Confirme se o índice FAISS existente está disponível localmente.")
        return 1
    except Exception as exc:  # falha operacional deve ser visível no terminal
        print(f"Falha na avaliação: {type(exc).__name__}: {exc}")
        return 1

    print("MÉDIAS POR ESTRATÉGIA")
    print(f"{'Estratégia':<14} {'Precision@4':>12} {'Recall@4':>10} {'MRR':>10}")
    for strategy in STRATEGIES:
        metrics = results["summary"][strategy]
        print(
            f"{strategy:<14} "
            f"{metrics['precision_at_k']:>11.3f} "
            f"{metrics['recall_at_k']:>10.3f} "
            f"{metrics['mrr']:>10.3f}"
        )

    print("\nVARIAÇÃO DA ESTRATÉGIA HÍBRIDA SOBRE A ORIGINAL")
    for metric, delta in results["hybrid_minus_original"].items():
        print(f"{metric}: {delta:+.3f} ({delta * 100:+.1f} p.p. quando aplicável)")

    json_path, csv_path = save_results(results)
    print("\nArquivos gerados:")
    print(f"- {json_path.relative_to(PROJECT_ROOT)}")
    print(f"- {csv_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
