
from typing import Any

from langchain_classic.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from business_agent.tools.sales_tool import sales_monthly_summary


class LegacyOpenAIToolsMockModel(FakeMessagesListChatModel):
    """Modelo determinístico para demonstrar o agente clássico com tool calling."""

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_results = [
            message
            for message in messages
            if isinstance(message, ToolMessage)
            or getattr(message, "type", None) == "function"
        ]

        if tool_results:
            response = AIMessage(
                content=(
                    "Análise de vendas executada com sucesso "
                    "pela ferramenta sales_monthly_summary.\n\n"
                    "Resultado SQL:\n"
                    f"{tool_results[-1].content}"
                )
            )
        else:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sales_monthly_summary",
                        "args": {"month": "2026-09"},
                        "id": "legacy-sales-monthly-001",
                    }
                ],
            )

        return ChatResult(
            generations=[
                ChatGeneration(message=response)
            ]
        )


def build_legacy_openai_tools_executor(
    model=None,
    tools=None,
    verbose=False,
):
    """Constrói create_openai_tools_agent + AgentExecutor.

    O agente aceita injeção de modelo para testes e para execução local com
    Ollama/Qwen3. O prompt orienta o uso explícito do histórico em follow-ups.
    """

    selected_tools = (
        tools if tools is not None else [sales_monthly_summary]
    )

    if not any(
        item.name == "sales_monthly_summary"
        for item in selected_tools
    ):
        raise ValueError(
            "A ferramenta sales_monthly_summary é obrigatória "
            "para esta demonstração."
        )

    selected_model = (
        model
        if model is not None
        else LegacyOpenAIToolsMockModel(responses=[])
    )

    available_tool_names = {
        item.name for item in selected_tools
    }

    routing_rules: list[str] = []

    if "sales_monthly_summary" in available_tool_names:
        routing_rules.append(
            "- Para receita, faturamento, descontos, pedidos e unidades "
            "em um período, utilize sales_monthly_summary."
        )

    if "sales_category_ranking" in available_tool_names:
        routing_rules.append(
            "- Para comparar categorias de produtos, utilize "
            "sales_category_ranking."
        )

    if "business_knowledge_search" in available_tool_names:
        routing_rules.extend(
            [
                "- Para definições de KPIs, fórmulas, nomes de campos, "
                "dicionário de dados e regras de negócio, utilize "
                "business_knowledge_search.",
                "- Não responda perguntas sobre campos e regras "
                "específicas do banco usando apenas conhecimento próprio. "
                "Consulte o RAG.",
            ]
        )

    if "search_business_memory" in available_tool_names:
        routing_rules.append(
            "- Para fatos, decisões, preferências ou análises registrados "
            "em sessões anteriores, utilize search_business_memory."
        )

    if "get_selic_target" in available_tool_names:
        routing_rules.append(
            "- Para taxa Selic atual, mais recente ou histórica, "
            "utilize get_selic_target."
        )

    few_shot_examples: list[str] = []

    if "business_knowledge_search" in available_tool_names:
        few_shot_examples.extend(
            [
                "EXEMPLO 1 - PERGUNTA CONCEITUAL:",
                "Pergunta: Como é calculada a receita líquida e quais são "
                "os campos físicos utilizados?",
                "Ação obrigatória: chamar business_knowledge_search.",
                "Consulta recomendada: receita líquida fórmula receita "
                "bruta descontos quantity unit_price discount_pct.",
                "Não invente nomes de colunas. Utilize o dicionário de dados.",
            ]
        )

    if (
        "sales_monthly_summary" in available_tool_names
        and "business_knowledge_search" in available_tool_names
    ):
        few_shot_examples.extend(
            [
                "",
                "EXEMPLO 2 - PERGUNTA COM DUAS FONTES:",
                "Pergunta: Qual foi a receita líquida de setembro de 2026 "
                "e como ela é calculada?",
                "Primeira ação: chamar sales_monthly_summary "
                "com month=2026-09.",
                "Segunda ação: chamar business_knowledge_search "
                "com a fórmula e os campos físicos da receita líquida.",
                "Resposta final: combinar os números do SQL com as "
                "definições documentadas no RAG.",
                "",
                "EXEMPLO 3 - FOLLOW-UP COM PERÍODO HERDADO:",
                "Histórico: o usuário perguntou pela receita líquida "
                "de setembro de 2026.",
                "Pergunta atual: E qual foi a receita bruta desse período?",
                "Ação obrigatória: recuperar setembro de 2026 do chat_history "
                "e chamar sales_monthly_summary com month=2026-09.",
                "Não responda usando apenas o histórico. Faça uma nova "
                "consulta SQL para a nova métrica.",
            ]
        )

    if "get_selic_target" in available_tool_names:
        few_shot_examples.extend(
            [
                "",
                "EXEMPLO 4 - API EXTERNA:",
                "Pergunta: Qual é a meta Selic mais recente?",
                "Ação obrigatória: chamar get_selic_target uma única vez, "
                "sem start_date/end_date, para obter a janela padrão.",
                "Use latest_observation.date e latest_observation.value "
                "do retorno dessa chamada.",
                "Não faça uma segunda consulta histórica se o usuário não "
                "solicitou explicitamente outro período.",
                "Não responda usando apenas conhecimento prévio.",
            ]
        )

    system_prompt = "\n".join(
        [
            "# PERSONA",
            "Você é um analista sênior de Business Intelligence.",
            "",
            "# TASK",
            "Responda perguntas de negócio utilizando as ferramentas "
            "disponíveis e as evidências retornadas.",
            "",
            "# CONTEXT",
            f"Ferramentas disponíveis: "
            f"{', '.join(sorted(available_tool_names))}",
            "",
            "# REGRAS OBRIGATÓRIAS DE ROTEAMENTO",
            *routing_rules,
            "",
            "# MEMÓRIA CONVERSACIONAL E FOLLOW-UPS",
            "- O chat_history contém mensagens anteriores desta mesma "
            "sessão; use-o para resolver referências como 'nesse mesmo mês', "
            "'desse período', 'esse resultado' e 'essa métrica'.",
            "- Se a pergunta atual solicitar uma nova métrica mensal e "
            "o mês estiver no chat_history, reutilize exatamente esse mês.",
            "- Para qualquer nova pergunta quantitativa sobre vendas, "
            "pedidos ou unidades, chame sales_monthly_summary novamente, "
            "mesmo que o mês tenha sido consultado no turno anterior.",
            "- Nunca responda uma pergunta quantitativa de vendas apenas "
            "copiando um número do histórico; execute a ferramenta SQL "
            "no turno atual.",
            "- Se o usuário pedir 'receita bruta desse período' após "
            "perguntar sobre setembro de 2026, chame "
            "sales_monthly_summary com month=2026-09.",
            "- Se não houver período identificável no histórico, peça "
            "esclarecimento em vez de inventar uma data.",
            "",
            "# MEMÓRIA PERSISTENTE",
            "- Quando a pergunta depender de fatos, decisões ou "
            "informações registrados anteriormente, consulte "
            "search_business_memory.",
            "- Use as memórias recuperadas como contexto histórico "
            "fornecido pelo usuário.",
            "- Não trate memórias históricas como valores atuais; "
            "para métricas atuais, consulte as ferramentas SQL.",
            "- Não afirme que uma memória foi salva; a gravação só "
            "ocorre quando o usuário solicita explicitamente.",
            "",
            "# REGRAS ESPECÍFICAS DA API SELIC (SGS 432)",
            "- Quando o usuário pedir a meta Selic mais recente, última observação "
            "ou taxa atual, chame get_selic_target uma única vez sem informar "
            "start_date/end_date.",
            "- Nessa consulta, use exclusivamente latest_observation.date e "
            "latest_observation.value do retorno da chamada.",
            "- Não faça uma segunda consulta para um mês ou intervalo histórico, "
            "a menos que o usuário tenha solicitado explicitamente esse período.",
            "- Não herde períodos de vendas, datas ou valores de memórias antigas "
            "para responder uma consulta da última observação Selic.",
            "- Para uma consulta histórica com período explicitamente informado, "
            "passe as datas solicitadas à ferramenta.",
            "",
            "# REGRAS PARA MÚLTIPLAS FONTES",
            "- Se a pergunta pedir um valor quantitativo e também sua "
            "definição, fórmula ou regra de negócio, consulte a ferramenta "
            "de dados e a ferramenta RAG.",
            "- Se a pergunta pedir somente um indicador ou valor, "
            "responda somente o que foi solicitado; não acrescente "
            "fórmulas ou definições não solicitadas.",
            "- Execute todas as consultas necessárias antes de formular "
            "a resposta final.",
            "- Uma resposta que exige duas fontes não está completa "
            "se apenas uma delas foi consultada.",
            "- Depois de receber o resultado de uma ferramenta, avalie "
            "se outra ferramenta ainda é necessária.",
            "",
            "# REGRAS DE CONFIABILIDADE",
            "- Não invente valores, campos, tabelas ou fontes.",
            "- Não apresente nomes de indicadores como se fossem "
            "colunas físicas do banco.",
            "- Utilize somente informações sustentadas pelos resultados "
            "das ferramentas para afirmações específicas do negócio.",
            "- Se uma ferramenta retornar erro ou não encontrar dados, "
            "informe a limitação.",
            "- Não substitua uma consulta externa por conhecimento próprio.",
            "- Não diga que consultou uma fonte se a ferramenta "
            "correspondente não foi executada no turno atual.",
            "",
            "# EXEMPLOS FEW-SHOT",
            *few_shot_examples,
            "",
            "# FORMAT",
            "Responda em português brasileiro, com clareza e objetividade.",
            "Informe períodos, unidades e fontes quando aplicável.",
            "Use concordância gramatical correta.",
            "Explique brevemente os cálculos somente quando solicitados, "
            "sem apresentar raciocínio interno privado.",
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(
                variable_name="chat_history",
                optional=True,
            ),
            ("human", "{input}"),
            MessagesPlaceholder(
                variable_name="agent_scratchpad",
            ),
        ]
    )

    agent = create_openai_tools_agent(
        selected_model,
        selected_tools,
        prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=selected_tools,
        verbose=verbose,
        return_intermediate_steps=True,
        max_iterations=4,
        handle_parsing_errors=True,
    )
