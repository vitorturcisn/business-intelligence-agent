from __future__ import annotations

import json
import math
import re
from typing import Any

from business_agent.verification.nli_verifier import (
    ClaimAssessment,
    Evidence,
    LocalNLIClassifier,
    NLIClassifier,
    VerificationReport,
    verify_response,
)


_RAG_BLOCK_SEPARATOR = re.compile(r"\r?\n\s*---\s*\r?\n")
_RAG_SOURCE_PATTERN = re.compile(r"(?m)^Fonte:[ \t]*(.+?)[ \t]*$")
_RAG_CONTENT_PATTERN = re.compile(r"(?ms)^Conteúdo:[ \t]*\r?\n(.*)$")
_MEMORY_TOOL_NAMES = {"persistent_memory_search", "search_business_memory"}


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def _split_rag_documents(content: str) -> list[Evidence]:
    """Split the formatted RAG response into one Evidence per source chunk."""
    documents: list[Evidence] = []

    for block in _RAG_BLOCK_SEPARATOR.split(content):
        source_match = _RAG_SOURCE_PATTERN.search(block)
        content_match = _RAG_CONTENT_PATTERN.search(block)

        if not source_match or not content_match:
            continue

        document_source = source_match.group(1).strip()
        document_content = content_match.group(1).strip()

        if not document_content:
            continue

        documents.append(
            Evidence(
                source=f"business_knowledge_search:{document_source}",
                content=document_content,
            )
        )

    return documents


def extract_executor_evidence(
    result: dict[str, Any],
) -> list[Evidence]:
    """Extract tool evidence, splitting RAG results by document/source."""
    evidence: list[Evidence] = []
    steps = result.get("intermediate_steps", [])

    # Memórias recuperadas ajudam a contextualizar a geração. Porém, quando
    # SQL/RAG/API também forneceu uma fonte específica para a resposta, não
    # devem competir como evidência factual daquela fonte.
    has_domain_tool_evidence = any(
        isinstance(step, (tuple, list))
        and len(step) == 2
        and str(getattr(step[0], "tool", "unknown_tool")) not in _MEMORY_TOOL_NAMES
        for step in steps
    )

    for step in steps:
        if not isinstance(step, (tuple, list)) or len(step) != 2:
            continue

        action, observation = step
        content = _to_text(observation).strip()

        if not content:
            continue

        source = str(getattr(action, "tool", "unknown_tool"))

        if source in _MEMORY_TOOL_NAMES and has_domain_tool_evidence:
            continue

        if source == "business_knowledge_search":
            rag_documents = _split_rag_documents(content)
            if rag_documents:
                evidence.extend(rag_documents)
                continue

        evidence.append(
            Evidence(
                source=source,
                content=content,
            )
        )

    return evidence



def _verify_text_to_query_exact_match(
    result: dict[str, Any],
) -> VerificationReport | None:
    """Verify Text-to-query output against its exact structured SQL result.

    This specialized path does not waive verification: it revalidates the
    allow-listed plan, validates the returned rows, reconstructs the canonical
    answer, and approves only an exact string match. Any mismatch falls back
    to the ordinary NLI verifier.
    """
    from business_agent.agents.text_to_query import (
        build_query_answer,
        parse_query_plan,
    )

    matching_steps = []
    for step in result.get("intermediate_steps", []):
        if not isinstance(step, (tuple, list)) or len(step) != 2:
            continue
        action, observation = step
        if getattr(action, "tool", None) == "text_to_query_sales":
            matching_steps.append(observation)

    if len(matching_steps) != 1:
        return None

    try:
        raw = matching_steps[0]
        payload = raw if isinstance(raw, dict) else json.loads(_to_text(raw))
        if not isinstance(payload, dict) or payload.get("status") != "executed":
            return None

        raw_plan = payload.get("plan")
        plan = parse_query_plan(json.dumps(raw_plan, ensure_ascii=False))

        period = payload.get("period")
        if period is not None and (
            not isinstance(period, str)
            or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", period)
        ):
            return None

        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) > plan["limit"]:
            return None

        for row in rows:
            if not isinstance(row, dict):
                return None
            dimension_value = row.get("dimension")
            metric_value = row.get("value")
            if not isinstance(dimension_value, str) or not dimension_value.strip():
                return None
            if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
                return None
            if not math.isfinite(float(metric_value)):
                return None

        expected_answer = build_query_answer(plan, rows, period)
        answer = result.get("output")
        if not isinstance(answer, str) or answer.strip() != expected_answer:
            return None

    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None

    assessment = ClaimAssessment(
        claim=answer.strip(),
        verdict="APOIA",
        confidence=1.0,
        evidence_source="text_to_query_sales",
        scores={"deterministic_exact_match": 1.0},
        unsupported_numbers=[],
        explanation=(
            "A resposta coincide exatamente com a formulação reconstruída "
            "a partir do plano validado e das linhas retornadas pelo SQL."
        ),
    )
    return VerificationReport(
        overall_verdict="SUSTENTADA",
        claims=[assessment],
        coherence_issues=[],
        evidence_sources=["text_to_query_sales"],
        summary=(
            "A resposta corresponde exatamente aos resultados da consulta "
            "SQL validada e executada em modo somente leitura."
        ),
    )


def verify_executor_result(
    result: dict[str, Any],
    classifier: NLIClassifier | None = None,
) -> VerificationReport:
    """Verify the final answer against observations returned by tools."""
    answer = result.get("output")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("O executor não retornou uma resposta válida.")

    text_to_query_report = _verify_text_to_query_exact_match(result)
    if text_to_query_report is not None:
        return text_to_query_report

    evidence = extract_executor_evidence(result)

    return verify_response(
        response=answer,
        evidence=evidence,
        classifier=classifier or LocalNLIClassifier(),
    )


def render_verified_answer(
    result: dict[str, Any],
    report: VerificationReport,
) -> str:
    """Release the original answer only when every claim is supported."""
    if report.overall_verdict == "SUSTENTADA":
        return result["output"]

    lines = [
        "RESPOSTA NÃO LIBERADA PELO VERIFICADOR.",
        "",
        f"Veredito: {report.overall_verdict}",
        report.summary,
        "",
        "Afirmações que precisam de revisão:",
    ]

    for assessment in report.claims:
        if assessment.verdict != "APOIA" or assessment.unsupported_numbers:
            lines.append(
                f"- [{assessment.verdict}] {assessment.claim}"
            )

            if assessment.unsupported_numbers:
                lines.append(
                    "  Valores sem correspondência nas evidências: "
                    + ", ".join(assessment.unsupported_numbers)
                )

    if report.coherence_issues:
        lines.extend(
            ["", "Foram identificadas possíveis incoerências entre afirmações."]
        )

    lines.extend(
        [
            "",
            "Fontes analisadas: "
            + (", ".join(report.evidence_sources) if report.evidence_sources else "nenhuma"),
        ]
    )

    return "\n".join(lines)
