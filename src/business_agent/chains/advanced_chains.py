
"""Chains sequenciais, paralelas e roteamento para o projeto.

O processamento é determinístico nesta etapa para permitir testes locais
sem chamadas a modelos externos.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_classic.chains.router.base import RouterChain


PERIOD_PATTERN = re.compile(r"\b(20\d{2}-(?:0[1-9]|1[0-2]))\b")


def _validate_question(inputs: dict[str, Any]) -> str:
    """Obtém e valida a pergunta recebida."""

    question = inputs.get("question", "")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Informe uma pergunta não vazia.")

    return question.strip()


def _classify_intent(question: str) -> str:
    """Classifica a intenção usando regras transparentes."""

    text = question.casefold()

    sales_terms = (
        "venda", "vendas", "faturamento", "receita", "categoria"
    )
    external_terms = ("selic", "banco central", "indicador econômico")
    definition_terms = (
        "definição", "definicao", "como calcular",
        "como calcula", "o que significa", "regra de cálculo",
    )

    has_sales = any(term in text for term in sales_terms)
    has_external = any(term in text for term in external_terms)
    asks_definition = any(term in text for term in definition_terms)

    if has_sales and has_external:
        return "multi_source_analysis"
    if asks_definition:
        return "metric_definition"
    if has_external:
        return "external_indicator"
    if has_sales:
        return "sales_analysis"

    return "other"


def _extract_periods(question: str) -> list[str]:
    """Extrai períodos explícitos no formato YYYY-MM."""

    return PERIOD_PATTERN.findall(question)


def _identify_sources(question: str) -> list[str]:
    """Identifica fontes potencialmente necessárias."""

    text = question.casefold()
    sources = []

    sales_terms = ("venda", "vendas", "faturamento", "receita", "categoria")
    external_terms = ("selic", "banco central", "indicador econômico")
    definition_terms = (
        "definição", "definicao", "como calcular",
        "como calcula", "o que significa", "regra de cálculo",
    )

    if any(term in text for term in sales_terms):
        sources.append("sql")

    if any(term in text for term in external_terms):
        sources.append("api_banco_central")

    if any(term in text for term in definition_terms):
        sources.append("rag")

    return sources


def _normalize(inputs: dict[str, Any]) -> dict[str, Any]:
    """Primeira etapa sequencial: normaliza a pergunta."""

    question = _validate_question(inputs)

    return {
        "question": question,
        "normalized_question": question.casefold(),
    }


def _extract_entities(inputs: dict[str, Any]) -> dict[str, Any]:
    """Segunda etapa sequencial: extrai períodos."""

    return {
        **inputs,
        "periods": _extract_periods(inputs["question"]),
    }


def _build_plan(inputs: dict[str, Any]) -> dict[str, Any]:
    """Terceira etapa sequencial: prepara um plano de consulta."""

    question = inputs["question"]
    intent = _classify_intent(question)
    sources = _identify_sources(question)

    return {
        **inputs,
        "intent": intent,
        "required_sources": sources,
        "steps": [
            f"Consultar a fonte: {source}"
            for source in sources
        ],
    }


# Chain sequencial: cada etapa recebe a saída da etapa anterior.
sequential_analysis_chain = (
    RunnableLambda(_normalize)
    | RunnableLambda(_extract_entities)
    | RunnableLambda(_build_plan)
)


# Chain paralela: cada branch processa a mesma pergunta independentemente.
parallel_analysis_chain = RunnableParallel(
    question=RunnableLambda(_validate_question),
    intent=RunnableLambda(
        lambda inputs: _classify_intent(
            _validate_question(inputs)
        )
    ),
    periods=RunnableLambda(
        lambda inputs: _extract_periods(
            _validate_question(inputs)
        )
    ),
    requested_sources=RunnableLambda(
        lambda inputs: _identify_sources(
            _validate_question(inputs)
        )
    ),
)



class BusinessIntentRouter(RouterChain):
    """RouterChain clássico com roteamento determinístico por regras."""

    @property
    def input_keys(self) -> list[str]:
        return ["question"]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Any = None,
    ) -> dict[str, Any]:
        """Seleciona o destino e devolve os dados para a próxima etapa."""

        question = _validate_question(inputs)
        destination = _classify_intent(question)

        return {
            "destination": destination,
            "next_inputs": {
                "question": question,
            },
        }


def build_router() -> BusinessIntentRouter:
    """Cria o roteador de intenções do domínio."""

    return BusinessIntentRouter()



def build_router() -> BusinessIntentRouter:
    """Cria o roteador de intenções do domínio."""

    return BusinessIntentRouter()
