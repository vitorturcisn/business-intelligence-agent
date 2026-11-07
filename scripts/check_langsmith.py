from dotenv import load_dotenv
from langsmith import Client

from business_agent.observability.langsmith_config import (
    get_langsmith_status,
)


def main():
    load_dotenv()

    status = get_langsmith_status()

    print("LANGSMITH API KEY:", "CONFIGURADA" if status.api_key_set else "AUSENTE")
    print("TRACING SOLICITADO:", status.tracing_requested)
    print("TRACING ATIVO:", status.tracing_enabled)
    print("PROJETO:", status.project)

    if not status.api_key_set:
        print("\nConfigure LANGSMITH_API_KEY no arquivo .env.")
        return

    try:
        client = Client()
        next(iter(client.list_projects(limit=1)), None)

        print("\nConexão com LangSmith validada.")

    except Exception as error:
        print("\nNão foi possível validar a conexão.")
        print("Tipo de erro:", type(error).__name__)
        print("Detalhe:", str(error))


if __name__ == "__main__":
    main()
