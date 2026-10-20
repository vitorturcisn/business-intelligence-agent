"""Verifica a disponibilidade das APIs exigidas na disciplina."""

def check_import(module_name, symbols):
    try:
        module = __import__(module_name, fromlist=symbols)

        for symbol in symbols:
            getattr(module, symbol)

        print(f"[OK] {module_name}: {', '.join(symbols)}")
        return True

    except Exception as exc:
        print(f"[ERRO] {module_name}: {type(exc).__name__}: {exc}")
        return False


def main():
    checks = [
        (
            "langchain_classic.agents",
            [
                "initialize_agent",
                "AgentType",
                "AgentExecutor",
                "create_openai_tools_agent",
            ],
        ),
        (
            "langchain_classic.memory",
            ["ConversationBufferMemory"],
        ),
        (
            "langchain_classic.chains.router.base",
            ["RouterChain"],
        ),
        (
            "langchain_core.runnables",
            ["Runnable", "chain"],
        ),
        (
            "langchain_core.runnables.history",
            ["RunnableWithMessageHistory"],
        ),
        (
            "langchain_openai",
            ["ChatOpenAI"],
        ),
        (
            "langchain_community.vectorstores",
            ["FAISS"],
        ),
    ]

    results = [check_import(module, symbols) for module, symbols in checks]

    print(f"\nResultado: {sum(results)}/{len(results)} verificações passaram.")

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()