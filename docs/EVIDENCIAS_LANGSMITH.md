# Evidências do LangSmith

**Projeto:** `business-intelligence-agent`  
**Dataset:** `business-intelligence-agent-qwen3-live-v1`  
**Experimento final:** `qwen3-live-agent-evaluation-37b531b2`  
**Data:** 08/11/2026  
**Modelo avaliado:** `qwen3:1.7b` local via Ollama  
**Tracing:** ativo na execução.

## Link da execução final

[ Abrir comparação/experimento no LangSmith ](https://smith.langchain.com/o/647ba4a7-e48d-499d-8d39-36bbf1042aeb/datasets/89d20358-ec83-4d57-91ad-e61482a5111a/compare?selectedSessions=b05e999c-94c3-40d1-bbb1-e626c6eea974)

## Resultado registrado

| Cenário | Veredito |
|---|---|
| `monthly_sql` | `SUSTENTADA` |
| `text_to_query` | `SUSTENTADA` |
| `external_api` | `SUSTENTADA` |
| `multi_turn_memory` | `SUSTENTADA` nos dois turnos |
| `rag_conceptual` | `SUSTENTADA` |
| `multi_source` | `SUSTENTADA` |

**Total:** 6/6 cenários sustentados.

## Artefato local correspondente

`outputs/qwen3_live_evaluation.json`

O JSON contém entradas, respostas, ferramentas utilizadas, status de verificação, afirmações avaliadas, fontes e escores NLI. Para a avaliação detalhada, consulte [`AVALIACAO_AGENTE_QWEN3.md`](AVALIACAO_AGENTE_QWEN3.md).

> Observação: o link do LangSmith pode exigir autenticação na organização/projeto. O arquivo JSON local foi incluído como evidência reprodutível complementar.


## Evidência complementar: avaliação mock anterior

As capturas históricas de 07/11/2026 documentam o experimento `business-agent-mock-evaluation-b393eeb1`, com cinco cenários e métricas `answer_coverage_evaluator` e `trajectory_evaluator` iguais a 1.00. Essa execução utilizou `RoutingEvaluationMockModel` (modelo determinístico/simulado) e serve como evidência complementar de roteamento e contratos de ferramentas. **Ela não substitui nem representa a avaliação ao vivo com Qwen3.**

As capturas estão catalogadas em [`EVIDENCIAS_LANGSMITH_MOCK.md`](EVIDENCIAS_LANGSMITH_MOCK.md) e reunidas em [`ANEXO_EVIDENCIAS_LANGSMITH_MOCK.pdf`](ANEXO_EVIDENCIAS_LANGSMITH_MOCK.pdf). A evidência principal da avaliação generativa real continua sendo o experimento `qwen3-live-agent-evaluation-37b531b2`, linkado acima.
