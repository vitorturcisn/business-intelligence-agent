"""Smoke test da ferramenta LangChain conectada à API real."""

import json

from business_agent.tools.bcb_api_tool import get_selic_target


def main():
    try:
        raw_result = get_selic_target.invoke({})
        result = json.loads(raw_result)
    except Exception as exc:
        raise SystemExit(
            f"Falha ao executar a ferramenta da API: {exc}"
        ) from exc

    if not result.get("observations"):
        raise SystemExit(
            "A ferramenta executou, mas não retornou observações."
        )

    print("Ferramenta LangChain e API real: OK")
    print(
        json.dumps(
            {
                "series_code": result["series_code"],
                "period": result["period"],
                "observations_count": result["observations_count"],
                "latest_observation": result["latest_observation"],
                "source": result["source"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()