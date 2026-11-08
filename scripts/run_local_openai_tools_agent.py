
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from langchain_core.agents import AgentAction
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from business_agent.agents.executor_verification import (
    render_verified_answer,
    verify_executor_result,
)
from business_agent.agents.legacy_openai_tools import (
    build_legacy_openai_tools_executor,
)
from business_agent.memory.short_term import (
    build_agent_with_session_history,
)
from business_agent.memory.persistent_memory import (
    PersistentMemoryManager,
    extract_explicit_memory_request,
    _is_personal_recall_query,
)
from business_agent.memory.tools import search_business_memory
from business_agent.tools.bcb_api_tool import get_selic_target
from business_agent.tools.rag_tool import business_knowledge_search
from business_agent.tools.sales_tool import (
    sales_monthly_summary as original_sales_monthly_summary,
)


MONTHS = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}

MONTH_NAMES = {
    number: name
    for name, number in MONTHS.items()
}

# Aliases usados pelo roteador determinístico de consultas mensais.
# Perguntas conceituais continuam sendo encaminhadas ao AgentExecutor/Qwen.
SALES_METRIC_ALIASES = {
    "net_revenue": (
        "receita liquida",
        "net revenue",
    ),
    "gross_revenue": (
        "receita bruta",
        "faturamento bruto",
        "gross revenue",
    ),
    "total_discounts": (
        "descontos totais",
        "desconto total",
        "total discounts",
    ),
    "order_count": (
        "numero de pedidos",
        "quantidade de pedidos",
        "total de pedidos",
        "pedidos",
        "pedido",
    ),
    "units_sold": (
        "unidades vendidas",
        "quantidade vendida",
        "total de unidades",
        "unidades",
        "unidade",
    ),
}

METRIC_FIELDS = {
    "net_revenue": "net_revenue",
    "gross_revenue": "gross_revenue",
    "total_discounts": "total_discounts",
    "order_count": "order_count",
    "units_sold": "units_sold",
}

CONCEPTUAL_MARKERS = (
    "diferenca entre",
    "diferenca",
    "o que e",
    "explique",
    "explicar",
    "formula",
    "como calcular",
    "como e calculad",
    "definicao",
    "defina",
    "quais campos",
    "quais colunas",
    "dicionario de dados",
)


def normalize_text(value: str) -> str:
    """Remove acentos e normaliza espaços para comparação de intenções."""

    normalized = unicodedata.normalize(
        "NFKD",
        value.lower(),
    )

    without_accents = "".join(
        char
        for char in normalized
        if not unicodedata.combining(char)
    )

    return re.sub(
        r"\s+",
        " ",
        without_accents,
    ).strip()


def normalize_month(value: str) -> str:
    """Normaliza formatos de mês para YYYY-MM."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "O mês deve ser informado como texto não vazio."
        )

    text = normalize_text(value)

    # Formato YYYY-MM ou YYYY-M.
    match = re.fullmatch(
        r"(\d{4})-(\d{1,2})",
        text,
    )

    if match:
        year, month = match.groups()
        month_number = int(month)

        if 1 <= month_number <= 12:
            return f"{year}-{month_number:02d}"

    # Formatos MM-YYYY ou MM/YYYY.
    match = re.fullmatch(
        r"(\d{1,2})[-/](\d{4})",
        text,
    )

    if match:
        month, year = match.groups()
        month_number = int(month)

        if 1 <= month_number <= 12:
            return f"{year}-{month_number:02d}"

    # Nome do mês + ano, por exemplo: setembro de 2026.
    match = re.fullmatch(
        r"([a-z]+)\s+(?:de\s+)?(\d{4})",
        text,
    )

    if match:
        month_name, year = match.groups()

        if month_name in MONTHS:
            return f"{year}-{MONTHS[month_name]:02d}"

    raise ValueError(
        f"Mês inválido: {value}. "
        "Utilize, por exemplo, 2026-09 ou setembro de 2026."
    )


def extract_month_from_text(
    text: str,
) -> str | None:
    """Extrai um período mensal explícito de uma pergunta ou mensagem."""

    normalized = normalize_text(text)

    numeric_patterns = (
        r"\b(20\d{2})-(0?[1-9]|1[0-2])\b",
        r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b",
    )

    for pattern in numeric_patterns:
        match = re.search(
            pattern,
            normalized,
        )

        if match:
            try:
                return normalize_month(match.group(0))
            except ValueError:
                pass

    month_pattern = (
        r"\b(janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|"
        r"setembro|outubro|novembro|dezembro)\s+(?:de\s+)?(20\d{2})\b"
    )

    match = re.search(
        month_pattern,
        normalized,
    )

    if match:
        return normalize_month(
            f"{match.group(1)} {match.group(2)}"
        )

    return None


def resolve_month(
    question: str,
    chat_history: list[Any],
) -> str | None:
    """Resolve o mês atual ou herda o último mês mencionado na sessão."""

    explicit_month = extract_month_from_text(question)

    if explicit_month:
        return explicit_month

    for message in reversed(chat_history or []):
        content = getattr(
            message,
            "content",
            "",
        )

        if isinstance(content, list):
            content = " ".join(
                str(part.get("text", ""))
                if isinstance(part, dict)
                else str(part)
                for part in content
            )

        inherited_month = extract_month_from_text(
            str(content)
        )

        if inherited_month:
            return inherited_month

    return None


def detect_sales_metric(
    question: str,
) -> str | None:
    """Identifica uma única métrica mensal quantitativa na pergunta."""

    normalized = normalize_text(question)

    if any(
        marker in normalized
        for marker in CONCEPTUAL_MARKERS
    ):
        return None

    found: list[tuple[int, str]] = []

    for metric, aliases in SALES_METRIC_ALIASES.items():
        for alias in aliases:
            if re.search(
                rf"(?<!\w){re.escape(alias)}(?!\w)",
                normalized,
            ):
                found.append(
                    (len(alias), metric)
                )

    distinct_metrics = {
        metric
        for _, metric in found
    }

    if len(distinct_metrics) != 1:
        return None

    # Um dos aliases pode ser um termo curto (ex.: "pedido").
    # Todos apontam para a mesma métrica e podem ser consolidados.
    return max(
        found,
        key=lambda item: item[0],
    )[1]


RAG_INTENT_MARKERS = (
    "diferenca entre",
    "diferenca",
    "explique",
    "explicar",
    "o que e",
    "definicao",
    "defina",
    "formula",
    "como calcular",
    "como e calculada",
    "como e calculado",
    "quais campos",
    "quais colunas",
    "dicionario de dados",
    "regra de negocio",
)

RAG_BUSINESS_TERMS = (
    "receita",
    "faturamento",
    "desconto",
    "pedido",
    "pedidos",
    "unidade",
    "unidades",
    "categoria",
    "kpi",
    "indicador",
    "vendas",
    "dados",
)


def is_conceptual_rag_query(question: str) -> bool:
    """Identify conceptual business questions that require knowledge-base evidence."""
    normalized = normalize_text(question)

    has_conceptual_intent = any(
        marker in normalized
        for marker in RAG_INTENT_MARKERS
    )
    has_business_topic = any(
        term in normalized
        for term in RAG_BUSINESS_TERMS
    )

    return has_conceptual_intent and has_business_topic


def format_brl(value: Any) -> str:
    """Formata um valor como moeda brasileira."""

    number = float(value)

    formatted = f"{number:,.2f}"

    formatted = (
        formatted
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    return f"R$ {formatted}"


def format_month_label(
    period: str,
) -> str:
    """Converte YYYY-MM para o nome do mês em português."""

    year, month = period.split("-")

    return (
        f"{MONTH_NAMES[int(month)]} de {year}"
    )


def parse_tool_payload(
    observation: Any,
) -> dict[str, Any]:
    """Converte a observação SQL em um dicionário, com validação básica."""

    if isinstance(observation, dict):
        payload = observation

    elif isinstance(observation, str):
        try:
            payload = json.loads(observation)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "A ferramenta SQL não retornou JSON válido; "
                "não é seguro montar a resposta."
            ) from exc

    else:
        raise TypeError(
            "Formato inesperado no retorno da ferramenta SQL."
        )

    if not isinstance(payload, dict):
        raise ValueError(
            "O retorno SQL não contém um objeto de métricas."
        )

    return payload


def build_metric_answer(
    metric: str,
    payload: dict[str, Any],
    period: str,
) -> str:
    """Monta uma resposta curta com base exclusivamente no campo SQL escolhido."""

    field = METRIC_FIELDS[metric]

    if field not in payload:
        raise KeyError(
            f"A consulta SQL não retornou o campo obrigatório {field!r}."
        )

    value = payload[field]
    month_label = format_month_label(period)

    if metric in {
        "net_revenue",
        "gross_revenue",
        "total_discounts",
    }:
        labels = {
            "net_revenue": "A receita líquida",
            "gross_revenue": "A receita bruta",
            "total_discounts": "Os descontos totais",
        }

        if metric == "total_discounts":
            return (
                f"{labels[metric]} de {month_label} "
                f"foram de {format_brl(value)}."
            )

        return (
            f"{labels[metric]} de {month_label} "
            f"foi de {format_brl(value)}."
        )

    if metric == "order_count":
        count = int(value)
        noun = "pedido" if count == 1 else "pedidos"

        return (
            f"Em {month_label}, foram registrados "
            f"{count} {noun}."
        )

    if metric == "units_sold":
        count = int(value)
        noun = "unidade" if count == 1 else "unidades"

        return (
            f"Em {month_label}, foram vendidas "
            f"{count} {noun}."
        )

    raise ValueError(
        f"Métrica não suportada: {metric}"
    )


@tool("sales_monthly_summary")
def sales_monthly_summary_adapter(
    month: str,
) -> str:
    """Consulta o resumo mensal de vendas no SQLite.

    Utilize para obter pedidos, unidades, faturamento bruto,
    descontos e receita líquida.

    Informe o mês no formato YYYY-MM.
    """

    normalized_month = normalize_month(month)

    print(
        f"\n[SQL ADAPTER] Recebido: {month}"
        f" | Normalizado: {normalized_month}"
    )

    return original_sales_monthly_summary.invoke(
        {"month": normalized_month}
    )


def main() -> None:
    print("=" * 75)
    print("BUSINESS INTELLIGENCE AGENT")
    print("Qwen3 local + AgentExecutor + SQL + RAG + Banco Central")
    print("Memória conversacional por sessão ativada")
    print("Roteador determinístico para métricas mensais ativado")
    print("Roteador determinístico para memória pessoal ativado")
    print("Roteador conceitual com recuperação RAG prévia ativado")
    print("=" * 75)

    model = ChatOpenAI(
        model="qwen3:1.7b",
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

    print("\nFerramentas disponibilizadas:")

    for selected_tool in tools:
        print(
            "-",
            selected_tool.name,
        )

    persistent_memory = PersistentMemoryManager()

    executor = build_legacy_openai_tools_executor(
        model=model,
        tools=tools,
        verbose=False,
    )

    def route_or_execute(
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Route personal recall, SQL metrics and conceptual RAG requests."""

        question = str(
            inputs.get("input", "")
        ).strip()

        chat_history = inputs.get(
            "chat_history",
            [],
        )

        # ---------------------------------------------------------
        # ROUTE 1: PERSONAL MEMORY RECALL
        # ---------------------------------------------------------
        # Personal-recall questions search only memories explicitly
        # saved by the user. They do not need the LLM to paraphrase them.
        if _is_personal_recall_query(question):
            retrieved_memories = persistent_memory.retrieve(
                question,
                limit=3,
            )

            print(
                "\n[MEMÓRIA DE LONGO PRAZO] Memórias explícitas recuperadas:",
                len(retrieved_memories),
            )

            memory_observation = json.dumps(
                retrieved_memories,
                ensure_ascii=False,
                default=str,
            )

            memory_contents = [
                str(memory.get("content", "")).strip()
                for memory in retrieved_memories
                if str(memory.get("content", "")).strip()
            ]

            if memory_contents:
                answer = "\n".join(memory_contents)
            else:
                answer = (
                    "Não encontrei uma memória explicitamente salva "
                    "que responda a essa pergunta."
                )

            memory_action = AgentAction(
                tool="persistent_memory_search",
                tool_input={"query": question},
                log=(
                    "PERSONAL_MEMORY_ROUTER: "
                    "pergunta de recordação pessoal identificada; "
                    "resposta construída diretamente a partir das "
                    "memórias explícitas recuperadas."
                ),
            )

            print(
                "\n[ROTEADOR DE MEMÓRIA] "
                "Pergunta de recordação pessoal identificada."
            )

            return {
                "input": question,
                "output": answer,
                "intermediate_steps": [
                    (memory_action, memory_observation)
                ],
            }

        # ---------------------------------------------------------
        # ROUTE 2: MONTHLY SALES METRICS
        # ---------------------------------------------------------
        # SQL remains the source of truth for current quantitative values.
        metric = detect_sales_metric(question)
        period = resolve_month(
            question,
            chat_history,
        )

        if metric and period:
            print(
                "\n[ROTEADOR DETERMINÍSTICO] "
                "Métrica mensal identificada: "
                f"{metric}; período resolvido: {period}."
            )

            observation = sales_monthly_summary_adapter.invoke(
                {"month": period}
            )

            payload = parse_tool_payload(observation)
            answer = build_metric_answer(
                metric,
                payload,
                period,
            )

            action = AgentAction(
                tool="sales_monthly_summary",
                tool_input={"month": period},
                log=(
                    "DETERMINISTIC_ROUTER: "
                    "a camada de orquestração identificou "
                    "uma consulta mensal e executou a ferramenta SQL "
                    "para evitar respostas quantitativas sem evidência."
                ),
            )

            return {
                "input": question,
                "output": answer,
                "intermediate_steps": [
                    (action, observation)
                ],
            }

        # ---------------------------------------------------------
        # ROUTE 3: CONCEPTUAL QUESTIONS -> RAG + QWEN3
        # ---------------------------------------------------------
        # Retrieve the business documentation before invoking the LLM.
        # Do not mix historical memory records into conceptual answers.
        if is_conceptual_rag_query(question):
            print(
                "\n[ROTEADOR CONCEITUAL] "
                "Pergunta de negócio identificada; consultando o RAG."
            )

            rag_observation = business_knowledge_search.invoke(
                {"query": question}
            )

            rag_action = AgentAction(
                tool="business_knowledge_search",
                tool_input={"query": question},
                log=(
                    "RAG_DETERMINISTIC_ROUTER: "
                    "consulta conceitual encaminhada diretamente "
                    "à base de conhecimento antes da geração da resposta."
                ),
            )

            if not isinstance(rag_observation, str):
                rag_observation = str(rag_observation)

            agent_question = (
                f"{question}\n\n"
                "[EVIDÊNCIAS RECUPERADAS PELA BASE DE CONHECIMENTO / RAG]\n"
                f"{rag_observation}\n\n"
                "[INSTRUÇÕES PARA A RESPOSTA]\n"
                "A consulta RAG já foi executada nesta interação. "
                "Responda com base no conteúdo documental apresentado. "
                "Não utilize memórias históricas de outras análises como "
                "evidência para esta explicação conceitual. "
                "Não acrescente valores, datas ou exemplos numéricos que "
                "não foram solicitados. "
                "Não repita a consulta RAG. Se a documentação não sustentar "
                "algum detalhe, informe essa limitação. "
                "Responda em português brasileiro, de forma breve e direta."
            )

            result = executor.invoke(
                {
                    "input": agent_question,
                    "chat_history": chat_history,
                }
            )

            result.setdefault(
                "intermediate_steps",
                [],
            )
            result["intermediate_steps"] = (
                [(rag_action, rag_observation)]
                + result["intermediate_steps"]
            )

            return result

        # ---------------------------------------------------------
        # ROUTE 4: GENERAL QUESTIONS -> HYBRID MEMORY + QWEN3
        # ---------------------------------------------------------
        # Ordinary context questions can use SQLite + FAISS retrieval.
        retrieved_memories = persistent_memory.retrieve(
            question,
            limit=3,
        )

        memory_context = persistent_memory.format_context(
            retrieved_memories
        )

        memory_steps = []
        agent_question = question

        if retrieved_memories:
            print(
                "\n[MEMÓRIA DE LONGO PRAZO] Memórias contextuais recuperadas:",
                len(retrieved_memories),
            )

        if memory_context:
            agent_question = question + memory_context

            memory_action = AgentAction(
                tool="persistent_memory_search",
                tool_input={"query": question},
                log=(
                    "PERSISTENT_MEMORY_ROUTER: "
                    "memórias contextuais recuperadas pela busca "
                    "híbrida SQLite + FAISS."
                ),
            )

            memory_observation = json.dumps(
                retrieved_memories,
                ensure_ascii=False,
                default=str,
            )

            memory_steps.append(
                (memory_action, memory_observation)
            )

        result = executor.invoke(
            {
                "input": agent_question,
                "chat_history": chat_history,
            }
        )

        # Include retrieved memories as traceable evidence for NLI.
        result.setdefault(
            "intermediate_steps",
            [],
        )
        result["intermediate_steps"] = (
            memory_steps + result["intermediate_steps"]
        )

        return result

    # O wrapper mantém o histórico de sessão tanto para o roteador
    # quanto para o AgentExecutor.
    orchestrated_agent = RunnableLambda(
        route_or_execute
    )

    agent_with_history, session_histories = (
        build_agent_with_session_history(
            orchestrated_agent,
            max_messages=8,
        )
    )

    session_id = "qwen3-local-demo"

    print("\nMemória conversacional ativada.")
    print("Identificador da sessão:", session_id)
    print("Janela de contexto: até 8 mensagens anteriores.")
    print(
        "Histórico em RAM: será perdido quando o programa "
        "for encerrado."
    )
    print(
        "Consultas quantitativas mensais usam "
        "roteamento SQL determinístico."
    )
    print(
        "Perguntas conceituais de negócio consultam o RAG antes "
        "da geração com Qwen3 + AgentExecutor."
    )
    print(
        "Consultas externas não conceituais usam Qwen3 + AgentExecutor."
    )
    print(
        "Memória pessoal explícita: recuperação determinística no SQLite."
    )
    print(
        "Memória contextual: recuperação híbrida SQLite + FAISS."
    )
    print(
        "Para salvar: comece com 'Lembre-se que...' "
        "ou 'Guarde na memória:...'."
    )

    print("\nDigite suas perguntas.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        question = input("Você: ").strip()

        if question.lower() in {
            "sair",
            "exit",
            "quit",
        }:
            break

        if not question:
            continue

        # A gravação persistente ocorre somente após um pedido explícito.
        memory_to_save = extract_explicit_memory_request(
            question
        )

        if memory_to_save:
            try:
                saved_memory = persistent_memory.save_explicit_memory(
                    memory_to_save
                )

                print(
                    "\n[MEMÓRIA DE LONGO PRAZO] "
                    "Informação salva no SQLite."
                )
                print("Tipo:", saved_memory["kind"])
                print("Chave:", saved_memory["memory_key"])
                print("Conteúdo:", saved_memory["content"])
                print(
                    "A informação poderá ser recuperada "
                    "em sessões futuras."
                )

            except Exception as exc:
                print("\nERRO AO SALVAR MEMÓRIA:")
                print(
                    f"{type(exc).__name__}: {exc}"
                )

            print("\n")
            continue

        print("\nExecutando agente...\n")

        try:
            result = agent_with_history.invoke(
                {"input": question},
                config={
                    "configurable": {
                        "session_id": session_id,
                    }
                },
            )

            print(
                "\nVerificando resposta contra as evidências..."
            )

            verification_report = verify_executor_result(
                result
            )

            print("\nRELATÓRIO NLI:")

            print(
                verification_report.model_dump_json(
                    indent=2,
                    ensure_ascii=False,
                )
            )

            steps = result.get(
                "intermediate_steps",
                [],
            )

            print("\n" + "=" * 75)
            print("TRAJETÓRIA DE EXECUÇÃO")
            print("=" * 75)

            if not steps:
                print("Nenhuma ferramenta foi utilizada.")

            else:
                for index, (
                    action,
                    observation,
                ) in enumerate(
                    steps,
                    start=1,
                ):
                    print(f"\nEtapa {index}")
                    print("Ferramenta:", action.tool)
                    print("Argumentos:", action.tool_input)

                    action_log = getattr(
                        action,
                        "log",
                        "",
                    )

                    if "PERSONAL_MEMORY_ROUTER" in action_log:
                        print(
                            "Orquestração: roteador determinístico "
                            "de memória pessoal"
                        )

                    elif "RAG_DETERMINISTIC_ROUTER" in action_log:
                        print(
                            "Orquestração: roteador conceitual + "
                            "RAG + Qwen3 AgentExecutor"
                        )

                    elif "DETERMINISTIC_ROUTER" in action_log:
                        print(
                            "Orquestração: roteador determinístico "
                            "de métricas"
                        )

                    elif "PERSISTENT_MEMORY_ROUTER" in action_log:
                        print(
                            "Orquestração: busca contextual "
                            "híbrida SQLite + FAISS"
                        )

                    else:
                        print(
                            "Orquestração: Qwen3 + AgentExecutor"
                        )

                    print("Resultado:", observation)

            print("\n" + "=" * 75)
            print("RESPOSTA FINAL")
            print("=" * 75)

            print(
                render_verified_answer(
                    result,
                    verification_report,
                )
            )

            history = session_histories.get(session_id)

            if history is not None:
                print(
                    "\n[MEMÓRIA] Mensagens armazenadas "
                    "nesta sessão:",
                    len(history.messages),
                )

        except Exception as exc:
            print("\nERRO DURANTE A EXECUÇÃO:")
            print(
                f"{type(exc).__name__}: {exc}"
            )

        print("\n")


if __name__ == "__main__":
    main()
