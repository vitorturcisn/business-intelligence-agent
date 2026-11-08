from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.agents import AgentAction
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langsmith import Client

from business_agent.agents.executor_verification import (
    verify_executor_result,
    render_verified_answer,
)
from business_agent.agents.legacy_openai_tools import (
    build_legacy_openai_tools_executor,
)
from business_agent.agents.query_routing import is_latest_selic_query
from business_agent.agents.selic_response import build_latest_selic_answer
from business_agent.agents.text_to_query import (
    execute_text_to_query,
    is_dimensional_sales_query,
)
from business_agent.memory.short_term import build_agent_with_session_history
from business_agent.memory.persistent_memory import PersistentMemoryManager
from business_agent.memory.tools import search_business_memory
from business_agent.tools.bcb_api_tool import get_selic_target
from business_agent.tools.rag_tool import business_knowledge_search

# Reuse the same routing helpers and SQL adapter as the interactive demo.
# This avoids creating a second, differently configured sales implementation.
from run_local_openai_tools_agent import (
    build_metric_answer,
    detect_sales_metric,
    is_conceptual_rag_query,
    parse_tool_payload,
    resolve_month,
    sales_monthly_summary_adapter,
)

from business_agent.observability.langsmith_config import get_langsmith_status


DATASET_NAME = "business-intelligence-agent-qwen3-live-v1"
PROJECT_NAME = "business-intelligence-agent"
MODEL_NAME = "qwen3:1.7b"
OUTPUT_PATH = Path("outputs/qwen3_live_evaluation.json")

EVALUATION_CASES = [
    {
        "inputs": {
            "scenario": "monthly_sql",
            "question": "Qual foi a receita líquida em setembro de 2026?",
        },
        "outputs": {
            "expected_tool_counts": {"sales_monthly_summary": 1},
            "required_terms": ["receita líquida", "setembro de 2026", "R$"],
        },
        "metadata": {"scenario": "monthly_sql", "source": "SQLite"},
    },
    {
        "inputs": {
            "scenario": "rag_conceptual",
            "question": (
                "Como é calculada a receita líquida e quais campos físicos "
                "da tabela sales são utilizados?"
            ),
        },
        "outputs": {
            "expected_tool_counts": {"business_knowledge_search": 1},
            "required_terms": [
                "receita líquida",
                "quantity",
                "unit_price",
                "discount_pct",
            ],
        },
        "metadata": {"scenario": "rag_conceptual", "source": "FAISS/RAG"},
    },
    {
        "inputs": {
            "scenario": "external_api",
            "question": "Qual é a meta Selic mais recente disponível?",
        },
        "outputs": {
            "expected_tool_counts": {"get_selic_target": 1},
            "required_terms": ["Selic", "SGS 432", "% a.a."],
        },
        "metadata": {"scenario": "external_api", "source": "BCB SGS 432"},
    },
    {
        "inputs": {
            "scenario": "multi_source",
            "question": (
                "Qual foi a receita líquida em setembro de 2026 e qual é "
                "a fórmula para calcular esse indicador?"
            ),
        },
        "outputs": {
            "expected_tool_counts": {
                "sales_monthly_summary": 1,
                "business_knowledge_search": 1,
            },
            "required_terms": [
                "receita líquida",
                "setembro de 2026",
                "discount_pct",
            ],
        },
        "metadata": {
            "scenario": "multi_source",
            "source": "SQLite + FAISS/RAG",
        },
    },
    {
        "inputs": {
            "scenario": "text_to_query",
            "question": (
                "Quais foram as três categorias com maior receita líquida "
                "em setembro de 2026?"
            ),
        },
        "outputs": {
            "expected_tool_counts": {"text_to_query_sales": 1},
            "required_terms": ["categoria", "R$"],
        },
        "metadata": {"scenario": "text_to_query", "source": "Qwen3 + SQL"},
    },
    {
        "inputs": {
            "scenario": "multi_turn_memory",
            "question": "Qual foi a receita líquida em setembro de 2026?",
            "follow_up_question": "E qual foi a receita bruta desse mesmo período?",
        },
        "outputs": {
            "expected_tool_counts": {"sales_monthly_summary": 2},
            "required_terms": [
                "receita líquida",
                "receita bruta",
                "setembro de 2026",
            ],
        },
        "metadata": {
            "scenario": "multi_turn_memory",
            "source": "SQLite + RunnableWithMessageHistory",
        },
    },
]


LIVE_RESULTS: list[dict[str, Any]] = []


def build_live_runtime():
    """Build the Qwen3 runtime using the same components as the CLI demo."""
    model = ChatOpenAI(
        model=MODEL_NAME,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
        max_retries=1,
        timeout=180,
    )

    tools = [
        sales_monthly_summary_adapter,
        business_knowledge_search,
        get_selic_target,
        search_business_memory,
    ]

    executor = build_legacy_openai_tools_executor(
        model=model,
        tools=tools,
        verbose=False,
    )
    persistent_memory = PersistentMemoryManager()

    def route_or_execute(inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs.get("input", "")).strip()
        chat_history = inputs.get("chat_history", [])

        # Same priority order as scripts/run_local_openai_tools_agent.py.
        if is_dimensional_sales_query(question):
            period = resolve_month(question, chat_history)
            try:
                query_result = execute_text_to_query(
                    question=question,
                    model=model,
                    period=period,
                )
                plan = query_result["plan"]
                observation = json.dumps(
                    {
                        "status": "executed",
                        "plan": plan,
                        "period": period,
                        "rows": query_result["rows"],
                    },
                    ensure_ascii=False,
                    default=str,
                )
                answer = query_result["answer"]
            except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                plan = {"question": question}
                observation = json.dumps(
                    {"status": "rejected", "reason": str(exc)},
                    ensure_ascii=False,
                )
                answer = (
                    "Não foi possível converter a pergunta em uma consulta "
                    "segura. Nenhum SQL foi executado. Motivo: " + str(exc)
                )

            action = AgentAction(
                tool="text_to_query_sales",
                tool_input=plan,
                log=(
                    "TEXT_TO_QUERY_ROUTER: plano estruturado gerado pelo "
                    "Qwen3, validado e compilado em SQL somente leitura."
                ),
            )
            return {
                "input": question,
                "output": answer,
                "intermediate_steps": [(action, observation)],
            }

        # Multi-source route: SQL provides the metric value; RAG provides its definition.
        # This route must run before the monthly-SQL-only and conceptual-RAG-only routes.
        if is_conceptual_rag_query(question):
            metric = detect_sales_metric(question, allow_conceptual=True)
            period = resolve_month(question, chat_history)
            if metric and period:
                sql_observation = sales_monthly_summary_adapter.invoke(
                    {"month": period}
                )
                rag_observation = business_knowledge_search.invoke(
                    {"query": question}
                )
                sql_action = AgentAction(
                    tool="sales_monthly_summary",
                    tool_input={"month": period},
                    log="MULTI_SOURCE_ROUTER: monthly SQL query executed.",
                )
                rag_action = AgentAction(
                    tool="business_knowledge_search",
                    tool_input={"query": question},
                    log="MULTI_SOURCE_ROUTER: knowledge-base retrieval executed.",
                )
                agent_question = (
                    f"{question}\n\n"
                    "[SOURCES ALREADY QUERIED IN THIS TURN]\n"
                    "Both SQL and RAG have already been executed. Do not call "
                    "sales_monthly_summary or business_knowledge_search again.\n\n"
                    "[SQL EVIDENCE - structured result]\n"
                    f"{sql_observation}\n\n"
                    "[RAG EVIDENCE - documented definitions and rules]\n"
                    f"{rag_observation}\n\n"
                    "[RESPONSE RULES]\n"
                    "Use SQL only for the numeric value and period. Use RAG for "
                    "the formula, business rules and physical field descriptions. "
                    "When describing unit_price, reproduce the data dictionary "
                    "wording exactly: 'preço unitário do produto vendido'. "
                    "Do not add paraphrases in parentheses or invent field "
                    "descriptions. If evidence is insufficient, say so. "
                    "Answer in Brazilian Portuguese."
                )
                result = executor.invoke(
                    {"input": agent_question, "chat_history": chat_history}
                )
                result.setdefault("intermediate_steps", [])
                result["intermediate_steps"] = (
                    [(sql_action, sql_observation), (rag_action, rag_observation)]
                    + result["intermediate_steps"]
                )
                return result

        metric = detect_sales_metric(question)
        period = resolve_month(question, chat_history)

        if metric and period:
            observation = sales_monthly_summary_adapter.invoke(
                {"month": period}
            )
            payload = parse_tool_payload(observation)
            answer = build_metric_answer(metric, payload, period)
            action = AgentAction(
                tool="sales_monthly_summary",
                tool_input={"month": period},
                log=(
                    "DETERMINISTIC_ROUTER: consulta mensal de vendas "
                    "executada diretamente no SQLite."
                ),
            )
            return {
                "input": question,
                "output": answer,
                "intermediate_steps": [(action, observation)],
            }

        if is_conceptual_rag_query(question):
            rag_observation = business_knowledge_search.invoke(
                {"query": question}
            )
            rag_action = AgentAction(
                tool="business_knowledge_search",
                tool_input={"query": question},
                log=(
                    "RAG_DETERMINISTIC_ROUTER: recuperação documental "
                    "executada antes da geração da resposta."
                ),
            )
            agent_question = (
                f"{question}\n\n"
                "[EVIDÊNCIAS RECUPERADAS PELA BASE DE CONHECIMENTO / RAG]\n"
                f"{rag_observation}\n\n"
                "[INSTRUÇÕES PARA A RESPOSTA]\n"
                "A consulta RAG já foi executada nesta interação. "
                "Responda com base nas evidências documentais fornecidas. "
                "Preserve literalmente as descrições dos campos físicos "
                "conforme aparecem no dicionário de dados. Para unit_price, "
                "use exatamente: preço unitário do produto vendido. "
                "Não acrescente paráfrases entre parênteses. "
                "Não repita a consulta RAG. Se a documentação não sustentar "
                "algum detalhe, informe a limitação. "
                "Responda em português brasileiro, de forma breve e direta."
            )
            result = executor.invoke(
                {"input": agent_question, "chat_history": chat_history}
            )
            result.setdefault("intermediate_steps", [])
            result["intermediate_steps"] = (
                [(rag_action, rag_observation)]
                + result["intermediate_steps"]
            )
            return result

        if is_latest_selic_query(question):
            api_question = (
                f"{question}\n\n"
                "[INSTRUÇÕES OBRIGATÓRIAS PARA A API SGS 432]\n"
                "Chame get_selic_target exatamente uma vez, sem informar "
                "start_date ou end_date. Use exclusivamente "
                "latest_observation.date e latest_observation.value. "
                "Informe a taxa em percentual ao ano."
            )
            result = executor.invoke(
                {"input": api_question, "chat_history": []}
            )
            result["output"] = build_latest_selic_answer(
                result.get("intermediate_steps", [])
            )
            return result

        retrieved_memories = persistent_memory.retrieve(question, limit=3)
        memory_context = persistent_memory.format_context(retrieved_memories)
        memory_steps = []
        agent_question = question

        if memory_context:
            agent_question = question + memory_context
            memory_action = AgentAction(
                tool="persistent_memory_search",
                tool_input={"query": question},
                log=(
                    "PERSISTENT_MEMORY_ROUTER: recuperação híbrida "
                    "SQLite + FAISS."
                ),
            )
            memory_observation = json.dumps(
                retrieved_memories,
                ensure_ascii=False,
                default=str,
            )
            memory_steps.append((memory_action, memory_observation))

        result = executor.invoke(
            {"input": agent_question, "chat_history": chat_history}
        )
        result.setdefault("intermediate_steps", [])
        result["intermediate_steps"] = (
            memory_steps + result["intermediate_steps"]
        )
        return result

    orchestrated_agent = RunnableLambda(route_or_execute)
    agent_with_history, histories = build_agent_with_session_history(
        orchestrated_agent,
        max_messages=8,
    )
    return agent_with_history, histories


def _tool_names(result: dict[str, Any]) -> list[str]:
    names = []
    for step in result.get("intermediate_steps", []):
        if isinstance(step, (tuple, list)) and len(step) == 2:
            action = step[0]
            name = getattr(action, "tool", None)
            if name:
                names.append(str(name))
    return names


def _run_turn(
    agent_with_history,
    question: str,
    session_id: str,
    scenario: str,
    turn_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = agent_with_history.invoke(
        {"input": question},
        config={
            "configurable": {"session_id": session_id},
            "run_name": f"qwen3-live-{scenario}-turn-{turn_number}",
            "tags": [
                "business-intelligence-agent",
                "qwen3-live-evaluation",
                scenario,
            ],
            "metadata": {
                "model_mode": "qwen3_local_real",
                "model_name": MODEL_NAME,
                "scenario": scenario,
                "turn_number": turn_number,
                "evaluation_dataset": DATASET_NAME,
            },
        },
    )
    report = verify_executor_result(result)
    return result, report.model_dump()


def target(inputs: dict[str, Any]) -> dict[str, Any]:
    scenario = str(inputs["scenario"])
    question = str(inputs["question"])
    session_id = f"qwen3-eval-{scenario}-{uuid.uuid4().hex}"

    turns = [(question, 1)]
    follow_up = inputs.get("follow_up_question")
    if follow_up:
        turns.append((str(follow_up), 2))

    answers: list[str] = []
    candidate_answers: list[str] = []
    statuses: list[str] = []
    summaries: list[str] = []
    verification_reports: list[dict[str, Any]] = []
    tools_used: list[str] = []

    for turn_question, turn_number in turns:
        try:
            result, report = _run_turn(
                AGENT_WITH_HISTORY,
                turn_question,
                session_id,
                scenario,
                turn_number,
            )
            candidate_answers.append(str(result.get("output", "")))
            answers.append(
                render_verified_answer(
                    result,
                    _report_from_dict(report),
                )
            )
            statuses.append(str(report["overall_verdict"]))
            summaries.append(str(report["summary"]))
            verification_reports.append(report)
            tools_used.extend(_tool_names(result))
        except Exception as exc:
            raw_error = f"ERRO NA EXECUÇÃO: {type(exc).__name__}: {exc}"
            candidate_answers.append(raw_error)
            answers.append(raw_error)
            statuses.append("ERRO")
            summaries.append(str(exc))
            verification_reports.append({"error": str(exc)})

    status = (
        "SUSTENTADA"
        if statuses and all(item == "SUSTENTADA" for item in statuses)
        else "NAO_SUSTENTADA"
    )
    output = {
        "scenario": scenario,
        "answer": "\n\n".join(answers),
        "candidate_answer": "\n\n".join(candidate_answers),
        "tool_names": tools_used,
        "tool_counts": dict(Counter(tools_used)),
        "verification_status": status,
        "verification_statuses": statuses,
        "verification_summaries": summaries,
        "verification_reports": verification_reports,
    }
    LIVE_RESULTS.append(
        {
            "inputs": inputs,
            "outputs": output,
        }
    )
    return output


def _report_from_dict(data: dict[str, Any]):
    # Reconstruct the Pydantic report so the same release guard used by the
    # interactive demo controls what the evaluator records as the answer.
    from business_agent.verification.nli_verifier import VerificationReport

    return VerificationReport.model_validate(data)


def _reference_outputs(
    reference_outputs: dict[str, Any] | None = None,
    example: Any = None,
) -> dict[str, Any]:
    candidate = reference_outputs
    if isinstance(candidate, dict) and isinstance(
        candidate.get("reference_outputs"), dict
    ):
        candidate = candidate["reference_outputs"]
    if not isinstance(candidate, dict) and example is not None:
        candidate = getattr(example, "outputs", None)
    if isinstance(candidate, dict) and isinstance(
        candidate.get("reference_outputs"), dict
    ):
        candidate = candidate["reference_outputs"]
    if not isinstance(candidate, dict):
        raise ValueError("Gabarito ausente no exemplo do LangSmith.")
    return candidate


def tool_trajectory_precision(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    example: Any = None,
) -> dict[str, Any]:
    reference = _reference_outputs(reference_outputs, example)
    expected = Counter(reference.get("expected_tool_counts", {}))
    actual = Counter(outputs.get("tool_names", []))
    true_positives = sum(min(actual[name], count) for name, count in expected.items())
    total_actual = sum(actual.values())
    score = true_positives / total_actual if total_actual else 0.0
    return {
        "score": score,
        "comment": (
            f"Chamadas esperadas: {dict(expected)}; "
            f"chamadas observadas: {dict(actual)}."
        ),
    }


def tool_trajectory_recall(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    example: Any = None,
) -> dict[str, Any]:
    reference = _reference_outputs(reference_outputs, example)
    expected = Counter(reference.get("expected_tool_counts", {}))
    actual = Counter(outputs.get("tool_names", []))
    total_expected = sum(expected.values())
    found = sum(min(actual[name], count) for name, count in expected.items())
    score = found / total_expected if total_expected else 1.0
    missing = {
        name: max(0, count - actual[name])
        for name, count in expected.items()
        if actual[name] < count
    }
    return {
        "score": score,
        "comment": f"Chamadas esperadas ausentes ou insuficientes: {missing or 'nenhuma'}.",
    }


def answer_coverage(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    example: Any = None,
) -> dict[str, Any]:
    reference = _reference_outputs(reference_outputs, example)
    answer = str(outputs.get("answer", "")).casefold()
    required = reference.get("required_terms", [])
    found = [term for term in required if str(term).casefold() in answer]
    missing = [term for term in required if str(term).casefold() not in answer]
    score = len(found) / len(required) if required else 1.0
    return {
        "score": score,
        "comment": f"Termos encontrados: {found}; ausentes: {missing}.",
    }


def verification_sustained(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    example: Any = None,
) -> dict[str, Any]:
    statuses = outputs.get("verification_statuses") or [
        outputs.get("verification_status", "ERRO")
    ]
    passed = bool(statuses) and all(item == "SUSTENTADA" for item in statuses)
    return {
        "score": passed,
        "comment": f"Vereditos do verificador: {statuses}.",
    }


def ensure_dataset(client: Client):
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset existente: {DATASET_NAME}")
    except Exception:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Avaliação do agente integrado com Qwen3 local real: "
                "SQL, RAG, API externa, Text-to-query e memória conversacional."
            ),
        )
        print(f"Dataset criado: {DATASET_NAME}")

    existing = list(client.list_examples(dataset_id=dataset.id))
    questions = {
        (
            example.inputs.get("scenario"),
            example.inputs.get("question"),
            example.inputs.get("follow_up_question"),
        )
        for example in existing
    }
    missing = [
        case for case in EVALUATION_CASES
        if (
            case["inputs"].get("scenario"),
            case["inputs"].get("question"),
            case["inputs"].get("follow_up_question"),
        ) not in questions
    ]

    if missing:
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[case["inputs"] for case in missing],
            outputs=[case["outputs"] for case in missing],
            metadata=[case["metadata"] for case in missing],
        )
        print(f"Novos cenários adicionados: {len(missing)}")
    else:
        print("Todos os cenários já existem no dataset.")

    return dataset


def main() -> None:
    load_dotenv()
    status = get_langsmith_status()
    print("LANGSMITH API KEY:", "CONFIGURADA" if status.api_key_set else "AUSENTE")
    print("TRACING ATIVO:", status.tracing_enabled)
    print("PROJETO:", status.project)

    if not status.api_key_set or not status.tracing_enabled:
        raise RuntimeError(
            "LangSmith não está pronto. Verifique LANGSMITH_API_KEY e "
            "LANGSMITH_TRACING=true no arquivo .env."
        )

    global AGENT_WITH_HISTORY
    AGENT_WITH_HISTORY, _ = build_live_runtime()

    client = Client()
    dataset = ensure_dataset(client)

    print("\nIniciando avaliação com Qwen3 local real.")
    print("Execução sequencial para reduzir consumo de memória.")
    print("Serão executados seis cenários, incluindo uma conversa de dois turnos.")
    print("O Qwen3 deve estar disponível no Ollama em localhost:11434.\n")

    experiment = client.evaluate(
        target,
        data=dataset.name,
        evaluators=[
            tool_trajectory_precision,
            tool_trajectory_recall,
            answer_coverage,
            verification_sustained,
        ],
        experiment_prefix="qwen3-live-agent-evaluation",
        description=(
            "Avaliação do runtime integrado real com Qwen3 1.7B local, "
            "AgentExecutor, SQL, RAG, API SGS 432, Text-to-query, "
            "RunnableWithMessageHistory e verificação NLI."
        ),
        max_concurrency=1,
        metadata={
            "model_mode": "qwen3_local_real",
            "model_name": MODEL_NAME,
            "langsmith_project": PROJECT_NAME,
            "agent_executor": "create_openai_tools_agent + AgentExecutor",
        },
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    local_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "langsmith_project": PROJECT_NAME,
        "model": MODEL_NAME,
        "evaluation_mode": "real_qwen3_local",
        "total_examples": len(LIVE_RESULTS),
        "results": LIVE_RESULTS,
        "langsmith_experiment": str(experiment),
    }
    OUTPUT_PATH.write_text(
        json.dumps(local_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nAvaliação concluída.")
    print("Dataset:", DATASET_NAME)
    print("Projeto LangSmith:", PROJECT_NAME)
    print("Cenários executados:", len(LIVE_RESULTS))
    print("Relatório local:", OUTPUT_PATH.resolve())
    print("\nExperimento LangSmith:")
    print(experiment)


if __name__ == "__main__":
    main()
