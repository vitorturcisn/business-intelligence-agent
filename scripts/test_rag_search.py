from business_agent.tools.rag_tool import (
    search_knowledge_base,
    transform_query,
)


questions = [
    "Como calcular a receita líquida das vendas?",
    "Qual é a diferença entre quantidade de unidades vendidas e número de pedidos?",
    "Qual é a fonte dos dados da taxa Selic e quais são as limitações?",
    "Como descobrir o faturamento após os descontos?",
    "De onde vêm os dados dos juros básicos da economia brasileira?",
]

for question in questions:
    print("\n" + "=" * 90)
    print(f"PERGUNTA ORIGINAL: {question}")
    print("=" * 90)

    transformed = transform_query(question)

    print("\nPERGUNTA TRANSFORMADA:")
    print(transformed)

    print("\nRESULTADOS DA RECUPERAÇÃO:")

    result = search_knowledge_base(question, k=4)

    print(result)
