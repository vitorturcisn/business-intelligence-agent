import json

from langchain_core.tools import tool


@tool
def get_selic_target(
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """
    Ferramenta simulada da API SGS do Banco Central.
    Utilizada exclusivamente nos testes de avaliação de trajetória.
    Não representa uma consulta real à API.
    """
    result = {
        "source": "Banco Central do Brasil",
        "system": "Sistema Gerenciador de Séries Temporais (SGS)",
        "series": 432,
        "start_date": start_date,
        "end_date": end_date,
        "fixture": True,
        "note": (
            "Resposta simulada para validação da escolha e "
            "execução da ferramenta."
        ),
    }

    return json.dumps(result, ensure_ascii=False)
