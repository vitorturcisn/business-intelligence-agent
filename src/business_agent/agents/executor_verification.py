import json
from typing import Any

from business_agent.verification.nli_verifier import (
    Evidence,
    LocalNLIClassifier,
    NLIClassifier,
    VerificationReport,
    verify_response,
)


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def extract_executor_evidence(
    result: dict[str, Any],
) -> list[Evidence]:
    """Extrai evidências das ferramentas executadas pelo AgentExecutor."""

    evidence = []

    for step in result.get("intermediate_steps", []):
        if not isinstance(step, (tuple, list)) or len(step) != 2:
            continue

        action, observation = step

        content = _to_text(observation).strip()
        if not content:
            continue

        source = getattr(action, "tool", "unknown_tool")

        evidence.append(
            Evidence(
                source=str(source),
                content=content,
            )
        )

    return evidence


def verify_executor_result(
    result: dict[str, Any],
    classifier: NLIClassifier | None = None,
) -> VerificationReport:
    """Verifica a resposta final contra as observações das ferramentas."""

    answer = result.get("output")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("O executor nao retornou uma resposta valida.")

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
    """
    Libera a resposta original somente quando todas as afirmacoes
    forem consideradas sustentadas.
    """

    if report.overall_verdict == "SUSTENTADA":
        return result["output"]

    lines = [
        "RESPOSTA NAO LIBERADA PELO VERIFICADOR.",
        "",
        f"Veredito: {report.overall_verdict}",
        report.summary,
        "",
        "Afirmações que precisam de revisão:",
    ]

    for assessment in report.claims:
        if (
            assessment.verdict != "APOIA"
            or assessment.unsupported_numbers
        ):
            lines.append(
                f"- [{assessment.verdict}] {assessment.claim}"
            )

            if assessment.unsupported_numbers:
                lines.append(
                    "  Valores sem correspondencia nas evidencias: "
                    + ", ".join(assessment.unsupported_numbers)
                )

    if report.coherence_issues:
        lines.append("")
        lines.append("Foram identificadas possíveis incoerências entre afirmações.")

    lines.append("")
    lines.append(
        "Fontes analisadas: "
        + (
            ", ".join(report.evidence_sources)
            if report.evidence_sources
            else "nenhuma"
        )
    )

    return "\n".join(lines)
