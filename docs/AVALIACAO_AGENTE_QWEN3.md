# Avaliação ao vivo do agente com Qwen3 e LangSmith

**Data:** 08/11/2026  
**Modelo:** `qwen3:1.7b` (Ollama local)  
**Modo:** `real_qwen3_local`  
**Dataset:** `business-intelligence-agent-qwen3-live-v1`  
**Projeto LangSmith:** `business-intelligence-agent`  
**Experimento:** `qwen3-live-agent-evaluation-37b531b2`  
**Resultado:** 6 de 6 cenários com veredito geral `SUSTENTADA`.

## 1. Objetivo

Validar o fluxo integrado do agente com um LLM local real, chamadas efetivas às ferramentas, verificação factual e tracing no LangSmith. A execução foi sequencial para reduzir o consumo de memória da máquina.

## 2. Cenários e resultados

| Cenário | Ferramenta(s) observada(s) | Resultado | Evidência resumida |
|---|---|---|---|
| `monthly_sql` | `sales_monthly_summary` | **SUSTENTADA** | Receita líquida de setembro/2026: R$ 25.250,00. |
| `text_to_query` | `text_to_query_sales` | **SUSTENTADA** | Ranking por receita líquida: Eletrônicos R$ 18.335,00; Móveis R$ 4.410,00; Acessórios R$ 2.505,00. |
| `external_api` | `get_selic_target` | **SUSTENTADA** | Retorno da série SGS 432: 13,75% a.a., observação de 08/11/2026. |
| `multi_turn_memory` | `sales_monthly_summary` (duas chamadas) | **SUSTENTADA** | Recuperou o mesmo período no follow-up; receita líquida R$ 25.250,00 e bruta R$ 26.650,00. |
| `rag_conceptual` | `business_knowledge_search` | **SUSTENTADA** | Fórmulas de receita bruta, desconto e receita líquida e definições dos campos físicos foram apoiadas pelo dicionário. |
| `multi_source` | `sales_monthly_summary` + `business_knowledge_search` | **SUSTENTADA** | Combinou valor SQL, fórmula documental, período e moeda. |

## 3. Resultado do RAG conceitual

O relatório de avaliação registrou sete afirmações atômicas, todas com veredito `APOIA`:

| Afirmação | Entailment |
|---|---:|
| Receita bruta por item: `quantity × unit_price` | 98,97% |
| Desconto por item: `quantity × unit_price × discount_pct` | 99,56% |
| Receita líquida por item: `quantity × unit_price × (1 - discount_pct)` | 82,57% |
| Definição de `quantity` | 98,19% |
| Definição de `unit_price` | 94,82% |
| Definição de `discount_pct` | 97,61% |
| Receita líquida = receita bruta − descontos (formulação do glossário) | 82,08% |

Os valores acima são escores do classificador NLI para pares específicos, não probabilidades calibradas. A saída geral do cenário foi `SUSTENTADA`.

## 4. Resultado multi-source

A resposta combinou `sales_monthly_summary` e `business_knowledge_search`. O conjunto de afirmações verificado incluiu receita líquida de R$ 25.250,00, identidade do KPI (receita bruta menos descontos), cálculo aritmético 26.650,00 − 1.400,00 = 25.250,00, período 2026-09, moeda BRL e a observação com os totais de receita bruta e descontos. Todas receberam `APOIA`.

## 5. Reprodutibilidade

Na raiz do projeto, com o ambiente virtual ativo, o Ollama disponível e LangSmith configurado:

```powershell
python scripts/evaluate_live_qwen_agent.py
```

O script executa seis cenários sequencialmente e grava `outputs/qwen3_live_evaluation.json`. A execução registrada neste documento foi gerada em `2026-11-08T21:59:49.873286+00:00` (UTC).

## 6. Evidência LangSmith

[ Abrir experimento final no LangSmith ](https://smith.langchain.com/o/647ba4a7-e48d-499d-8d39-36bbf1042aeb/datasets/89d20358-ec83-4d57-91ad-e61482a5111a/compare?selectedSessions=b05e999c-94c3-40d1-bbb1-e626c6eea974)

- Experiment: `qwen3-live-agent-evaluation-37b531b2`
- Dataset: `business-intelligence-agent-qwen3-live-v1`
- Cenários executados: 6
- Tracing ativo: sim, conforme log do terminal.

## 7. Limitações

- São seis cenários controlados, selecionados para cobrir rotas principais; não representam todas as consultas possíveis.
- O resultado é específico ao modelo `qwen3:1.7b`, ao prompt, às ferramentas, ao dataset e ao ambiente local testados.
- O teste multi-turn cobre uma conversa curta de dois turnos, não persistência de memória após reinicialização do processo.
- O sucesso do verificador não é garantia absoluta de correção; o NLI é uma camada auxiliar e seus escores não são probabilidades calibradas.
- A avaliação deve ser interpretada junto aos testes unitários e ao benchmark de recuperação RAG, não como métrica única de qualidade.
