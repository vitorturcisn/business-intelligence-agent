# Evidências históricas do avaliador mock no LangSmith

**Experimento:** `business-agent-mock-evaluation-b393eeb1`  
**Data das capturas:** 07/11/2026  
**Quantidade:** 5 cenários  
**Tipo de execução:** modelo determinístico/simulado (`RoutingEvaluationMockModel`)

## Interpretação correta

Estas capturas são evidências históricas de uma avaliação de integração e roteamento com modelo mock. A tabela do experimento mostra cinco execuções e médias `1.00` para `answer_coverage_evaluator` e `trajectory_evaluator`. As capturas individuais mostram o rastreamento das ferramentas SQL, RAG e API externa.

**Importante:** esta execução não é a avaliação ao vivo com Qwen3 e não deve ser apresentada como evidência de inferência generativa real. A evidência final da avaliação real está em [`EVIDENCIAS_LANGSMITH.md`](EVIDENCIAS_LANGSMITH.md), no experimento `qwen3-live-agent-evaluation-37b531b2`, com seis cenários e veredito `SUSTENTADA`.

## Capturas

### Resumo do experimento mock

![Resumo do experimento mock com cinco execuções e métricas 1.00](evidencias_langsmith_mock/01_resumo_avaliacao_mock.png)

### Traço: ranking de categorias

![Detalhe do traço de ranking de categorias](evidencias_langsmith_mock/02_trace_categorias.png)

### Traço: consulta multi-source

![Detalhe do traço que consulta SQL e RAG](evidencias_langsmith_mock/03_trace_multi_source.png)

### Traço: recuperação RAG

![Detalhe do traço de consulta documental](evidencias_langsmith_mock/04_trace_rag.png)

### Traço: consulta SQL mensal

![Detalhe do traço de consulta SQL mensal](evidencias_langsmith_mock/05_trace_sql.png)

### Traço: API externa

![Detalhe do traço da API Selic](evidencias_langsmith_mock/06_trace_api.png)
