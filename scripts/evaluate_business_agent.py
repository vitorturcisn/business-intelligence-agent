import argparse
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client

from business_agent.agents.orchestrator import build_business_agent
from business_agent.agents.evaluation_model import RoutingEvaluationMockModel
from business_agent.memory.tools import get_business_memory
from business_agent.tools.sales_tool import (
    sales_monthly_summary,
    sales_category_ranking,
)
from business_agent.tools.rag_tool import business_knowledge_search
from business_agent.tools.evaluation_tools import get_selic_target
from business_agent.observability.langsmith_config import (
    get_langsmith_status,
)


DATASET_NAME = "business-intelligence-agent-evaluation-v1"


EVALUATION_EXAMPLES = [
    {
        "inputs": {
            "question": (
                "Qual foi a receita líquida e quantos pedidos "
                "ocorreram em setembro de 2026?"
            )
        },
        "reference_outputs": {
            "expected_tools": ["sales_monthly_summary"],
            "required_terms": ["25250", "6"],
        },
        "metadata": {"scenario": "sql_sales"},
    },
    {
        "inputs": {
            "question": (
                "Quais categorias tiveram melhor desempenho "
                "em setembro de 2026?"
            )
        },
        "reference_outputs": {
            "expected_tools": ["sales_category_ranking"],
            "required_terms": ["2026-09"],
        },
        "metadata": {"scenario": "sql_category_ranking"},
    },
    {
        "inputs": {
            "question": "Qual é a definição de receita líquida?"
        },
        "reference_outputs": {
            "expected_tools": ["business_knowledge_search"],
            "required_terms": ["receita líquida", "descontos"],
        },
        "metadata": {"scenario": "rag_definition"},
    },
    {
        "inputs": {
            "question": "Qual é a fonte de dados utilizada para a taxa Selic?"
        },
        "reference_outputs": {
            "expected_tools": ["get_selic_target"],
            "required_terms": ["Banco Central", "SGS", "432"],
        },
        "metadata": {"scenario": "external_api"},
    },
    {
        "inputs": {
            "question": (
                "Qual foi a receita líquida em setembro de 2026 "
                "e como esse indicador é calculado?"
            )
        },
        "reference_outputs": {
            "expected_tools": [
                "sales_monthly_summary",
                "business_knowledge_search",
            ],
            "required_terms": [
                "25250",
                "receita líquida",
                "descontos",
            ],
        },
        "metadata": {"scenario": "multi_source"},
    },
]


def build_evaluation_agent():
    tools = [
        sales_monthly_summary,
        sales_category_ranking,
        business_knowledge_search,
        get_selic_target,
    ]

    return build_business_agent(
        model=RoutingEvaluationMockModel(responses=[]),
        checkpointer=InMemorySaver(),
        tools=tools,
    )


def target(inputs: dict) -> dict:
    question = inputs["question"]

    agent = build_evaluation_agent()

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={
            "configurable": {
                "thread_id": f"evaluation-{uuid.uuid4().hex}",
            },
            "tags": [
                "business-intelligence-agent",
                "evaluation",
                "deterministic-mock",
            ],
            "metadata": {
                "model_mode": "deterministic_mock",
                "evaluation_dataset": DATASET_NAME,
            },
        },
    )

    messages = result.get("messages", [])

    tool_names = [
        message.name
        for message in messages
        if isinstance(message, ToolMessage)
    ]

    answer = str(messages[-1].content) if messages else ""

    return {
        "answer": answer,
        "tool_names": tool_names,
    }


def trajectory_evaluator(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    expected = set(reference_outputs["expected_tools"])
    actual = set(outputs.get("tool_names", []))

    score = expected == actual

    return {
        "score": score,
        "comment": (
            f"Esperadas: {sorted(expected)} | "
            f"Executadas: {sorted(actual)}"
        ),
    }


def answer_coverage_evaluator(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    answer = outputs.get("answer", "").casefold()

    required_terms = reference_outputs.get("required_terms", [])

    missing = [
        term for term in required_terms
        if term.casefold() not in answer
    ]

    score = len(missing) == 0

    return {
        "score": score,
        "comment": (
            "Todos os termos esperados foram encontrados."
            if score
            else f"Termos ausentes: {missing}"
        ),
    }


def run_local_evaluation():
    results = []

    print("\nEXECUTANDO AVALIAÇÃO LOCAL")
    print("=" * 80)

    for index, example in enumerate(EVALUATION_EXAMPLES, start=1):
        outputs = target(example["inputs"])
        reference = example["reference_outputs"]

        trajectory = trajectory_evaluator(outputs, reference)
        coverage = answer_coverage_evaluator(outputs, reference)

        result = {
            "scenario": example["metadata"]["scenario"],
            "question": example["inputs"]["question"],
            "expected_tools": reference["expected_tools"],
            "actual_tools": outputs["tool_names"],
            "trajectory_score": trajectory["score"],
            "answer_coverage_score": coverage["score"],
            "answer": outputs["answer"],
            "trajectory_comment": trajectory["comment"],
            "coverage_comment": coverage["comment"],
        }

        results.append(result)

        print(f"\nCENÁRIO {index}: {result['scenario']}")
        print("Ferramentas esperadas:", result["expected_tools"])
        print("Ferramentas executadas:", result["actual_tools"])
        print("Trajetória:", result["trajectory_score"])
        print("Cobertura da resposta:", result["answer_coverage_score"])
        print("Diagnóstico:", result["coverage_comment"])

    total = len(results)

    trajectory_rate = sum(
        item["trajectory_score"] for item in results
    ) / total

    coverage_rate = sum(
        item["answer_coverage_score"] for item in results
    ) / total

    summary = {
        "dataset": DATASET_NAME,
        "evaluation_mode": "deterministic_mock",
        "total_examples": total,
        "trajectory_accuracy": trajectory_rate,
        "answer_coverage": coverage_rate,
        "results": results,
    }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "business_agent_evaluation.json"

    output_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("RESUMO DA AVALIAÇÃO LOCAL")
    print("=" * 80)
    print(f"Casos avaliados: {total}")
    print(f"Precisão da trajetória: {trajectory_rate:.0%}")
    print(f"Cobertura das respostas: {coverage_rate:.0%}")
    print(f"Relatório salvo em: {output_file.resolve()}")

    return summary


def upload_to_langsmith():
    status = get_langsmith_status()

    if not status.api_key_set:
        raise RuntimeError(
            "LANGSMITH_API_KEY não configurada no .env."
        )

    client = Client()

    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset existente encontrado: {DATASET_NAME}")

    except Exception:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Avaliação de trajetória e cobertura de respostas "
                "do Business Intelligence Agent."
            ),
        )
        print(f"Dataset criado: {DATASET_NAME}")

    existing_examples = list(
        client.list_examples(dataset_id=dataset.id)
    )

    existing_questions = {
        example.inputs.get("question")
        for example in existing_examples
    }

    new_examples = [
        example
        for example in EVALUATION_EXAMPLES
        if example["inputs"]["question"] not in existing_questions
    ]

    if new_examples:
        client.create_examples(
            dataset_id=dataset.id,
            examples=new_examples,
        )
        print(f"Novos exemplos enviados: {len(new_examples)}")
    else:
        print("Os exemplos já existem no dataset.")

    print("\nEXECUTANDO EXPERIMENTO REMOTO...")

    experiment = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            trajectory_evaluator,
            answer_coverage_evaluator,
        ],
        experiment_prefix="business-agent-mock-evaluation",
        description=(
            "Avaliação de tool trajectory e answer coverage. "
            "O modelo utilizado é determinístico e simulado."
        ),
        max_concurrency=1,
        metadata={
            "model_mode": "deterministic_mock",
            "project": "business-intelligence-agent",
        },
    )

    print("\nEXPERIMENTO ENVIADO AO LANGSMITH:")
    print(experiment)

    if hasattr(client, "flush"):
        client.flush()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--upload",
        action="store_true",
        help="Envia o dataset e executa a avaliação no LangSmith.",
    )

    args = parser.parse_args()

    load_dotenv()

    run_local_evaluation()

    if args.upload:
        upload_to_langsmith()
    else:
        print(
            "\nModo local: nada foi enviado ao LangSmith. "
            "Use --upload depois de configurar sua API key."
        )


if __name__ == "__main__":
    main()
