from langgraph.checkpoint.memory import InMemorySaver

from business_agent.agents.orchestrator import build_business_agent
from business_agent.agents.mock_model import FakeSalesToolCallingModel
from business_agent.agents.runner import run_business_query
from business_agent.memory.long_term import LongTermMemoryStore


def build_mock_agent():
    return build_business_agent(
        model=FakeSalesToolCallingModel(responses=[]),
        checkpointer=InMemorySaver(),
    )


def main():
    store = LongTermMemoryStore()

    print("=" * 80)
    print("SESSÃO 1 — CONSULTA E REGISTRO DO EPISÓDIO")
    print("=" * 80)

    agent_session_1 = build_mock_agent()

    first = run_business_query(
        agent=agent_session_1,
        question="Consulte as vendas de setembro de 2026.",
        thread_id="session-persistent-001",
        memory_store=store,
        persist_episode=True,
    )

    print("\nFerramentas executadas:", first.tool_names)
    print("\nResposta:")
    print(first.agent_result["messages"][-1].content)
    print("\nEpisódio registrado:", first.saved_episode_key)

    print("\n" + "=" * 80)
    print("SESSÃO 2 — RECUPERAÇÃO DE MEMÓRIA PERSISTENTE")
    print("=" * 80)

    # Nova instância do agente e novo checkpoint de conversa.
    agent_session_2 = build_mock_agent()

    # Nova instância do armazenamento SQLite.
    reopened_store = LongTermMemoryStore()

    second = run_business_query(
        agent=agent_session_2,
        question="Retome a análise de vendas de setembro de 2026.",
        thread_id="session-persistent-002",
        memory_store=reopened_store,
        persist_episode=False,
    )

    print("\nMemórias recuperadas antes da consulta:")

    for memory in second.retrieved_memories:
        print(f"\nTipo: {memory['kind']}")
        print(f"Chave: {memory['memory_key']}")
        print(f"Conteúdo: {memory['content'][:500]}")

    print("\nFerramentas executadas na segunda sessão:")
    print(second.tool_names)

    print("\nResposta:")
    print(second.agent_result["messages"][-1].content)

    print("\n" + "=" * 80)
    print("VALIDAÇÃO")
    print("=" * 80)

    recovered = any(
        memory["memory_key"] == first.saved_episode_key
        for memory in second.retrieved_memories
    )

    print("Episódio da sessão 1 recuperado na sessão 2:", recovered)

    if not recovered:
        raise RuntimeError(
            "A segunda sessão não recuperou o episódio da primeira."
        )


if __name__ == "__main__":
    main()
