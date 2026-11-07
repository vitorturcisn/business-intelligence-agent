from business_agent.verification.nli_verifier import (
    Evidence,
    verify_response,
)


def main():
    evidence = [
        Evidence(
            source="sales_sql",
            content=(
                "Resumo de vendas de setembro de 2026: "
                "6 pedidos, 40 unidades vendidas, "
                "receita bruta de R$ 26.650,00, "
                "descontos de R$ 1.400,00 e "
                "receita líquida de R$ 25.250,00."
            ),
        ),
        Evidence(
            source="01_glossario_kpis.md",
            content=(
                "Receita líquida é o valor das vendas após os descontos. "
                "A fórmula é receita bruta menos descontos."
            ),
        ),
    ]

    response = (
        "Em setembro de 2026, foram registrados 6 pedidos. "
        "A receita líquida foi de R$ 25.250,00. "
        "Em setembro de 2026, foram registrados 12 pedidos. "
        "A receita líquida foi de R$ 30.000,00."
    )

    print("RESPOSTA SUBMETIDA AO VERIFICADOR:")
    print(response)

    print("\nCARREGANDO MODELO NLI LOCAL...")
    report = verify_response(
        response=response,
        evidence=evidence,
    )

    print("\nRELATÓRIO DE VERIFICAÇÃO:")
    print(report.model_dump_json(indent=2))

    print("\nRESULTADO GERAL:", report.overall_verdict)


if __name__ == "__main__":
    main()
