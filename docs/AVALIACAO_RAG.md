# Avaliação de recuperação RAG

**Data da execução:** 08/11/2026 17:52 UTC  
**Índice utilizado:** `data/indexes/business_knowledge_faiss`  
**Índice reconstruído:** Não  
**Perguntas:** 14  
**Top-k:** 4  
**Nível da avaliação:** documento/fonte; chunks repetidos da mesma fonte são contabilizados uma única vez.

## 1. Objetivo

Avaliar se a expansão determinística de consultas reduz falhas de recuperação causadas por diferenças de vocabulário entre a pergunta do usuário e os documentos da base de conhecimento. Foram comparadas três estratégias:

- **Original:** pesquisa apenas com a pergunta original.
- **Expanded:** pesquisa apenas com a consulta produzida por `transform_query` (quando há expansão; caso contrário, mantém a pergunta original).
- **Hybrid:** combina candidatos da pesquisa original e da consulta expandida, elimina duplicações no nível do chunk e ordena os candidatos pela menor distância L2 encontrada.

A avaliação reutilizou o índice FAISS existente. O índice não foi reconstruído ou alterado.

## 2. Métricas

- **Precision@4:** proporção de fontes relevantes entre as quatro posições consideradas. Fontes repetidas são contabilizadas uma vez.
- **Recall@4:** proporção das fontes relevantes esperadas que foram recuperadas nas quatro posições.
- **MRR (Mean Reciprocal Rank):** média do inverso da posição da primeira fonte relevante; vale zero quando nenhuma fonte relevante é recuperada.

A relevância foi definida por pergunta por meio de uma lista de fontes esperadas (`gold_sources`). Portanto, as métricas avaliam recuperação de documentos/fontes, não a correção factual da resposta gerada pelo LLM.

## 3. Resultados agregados

| Estratégia | Precision@4 | Recall@4 | MRR |
|---|---:|---:|---:|
| Original | 0,4643 | 0,9762 | 0,9643 |
| Expanded | 0,4821 | 1,0000 | 1,0000 |
| Hybrid | 0,4821 | 1,0000 | 1,0000 |

Variação da estratégia híbrida em relação à original:

| Métrica | Variação absoluta | Variação em pontos percentuais |
|---|---:|---:|
| Precision@4 | +0,0179 | +1,79 p.p. |
| Recall@4 | +0,0238 | +2,38 p.p. |
| MRR | +0,0357 | +3,57 p.p. |

Neste conjunto de 14 perguntas, a estratégia expandida e a híbrida obtiveram os mesmos resultados agregados e os mesmos rankings por fonte. Isso indica que a combinação das duas pesquisas não trouxe ganho adicional mensurável em relação à pesquisa expandida isoladamente neste benchmark.

## 4. Análise por pergunta

### RAG-02 — “Como descobrir o faturamento após os descontos?”

- **Original:** recuperou `01_glossario_kpis.md` e `02_dicionario_dados.md`; não recuperou `03_regras_negocio.md`, também marcada como relevante.
- **Expanded/Hybrid:** recuperaram as três fontes esperadas.
- **Precision@4:** 0,50 → 0,75.
- **Recall@4:** 0,6667 → 1,0000.
- **MRR:** permaneceu em 1,0000.

Este é o caso mais direto de correção de incompatibilidade vocabular: a pergunta usa “faturamento após os descontos”, enquanto a base formaliza o conceito como “receita líquida”. A expansão adicionou termos como receita líquida, receita bruta, descontos, `quantity`, `unit_price` e `discount_pct`.

### RAG-11 — “Qual é a série do Banco Central usada para consultar a taxa Selic?”

- **Original:** `03_regras_negocio.md` apareceu antes de `04_fontes_e_limitacoes.md`, a fonte esperada.
- **Expanded/Hybrid:** `04_fontes_e_limitacoes.md` passou para a primeira posição.
- **Recall@4:** permaneceu em 1,0000.
- **MRR:** 0,5000 → 1,0000.

A expansão aproximou a consulta da terminologia documentada — Banco Central, SGS e série 432 — e melhorou a posição da fonte correta.

### RAG-03 — “Qual é a fórmula da receita líquida para cada item vendido?”

A ordem das duas fontes relevantes foi invertida: o dicionário de dados passou à primeira posição. Como ambas já eram recuperadas na busca original, Precision@4, Recall@4 e MRR permaneceram iguais.

### Demais perguntas

Nas outras 11 perguntas, não houve alteração nas métricas agregadas por pergunta. As fontes relevantes continuaram sendo recuperadas. Isso mostra que, neste conjunto, a expansão não prejudicou as consultas avaliadas, mas também não trouxe melhoria mensurável para a maioria delas.

## 5. Interpretação

Os resultados sustentam que a expansão determinística ajudou em dois cenários específicos:

1. **Cobertura:** recuperou uma fonte adicional relevante para a pergunta sobre faturamento após descontos.
2. **Ordenação:** colocou a fonte correta em primeiro lugar para a pergunta sobre a série do Banco Central usada na consulta da Selic.

O ganho médio é positivo, mas concentrado nesses dois casos. Não deve ser interpretado como uma melhoria universal do RAG. A igualdade entre `expanded` e `hybrid` também sugere que o custo de executar duas buscas pode não se justificar para este conjunto de avaliação; qualquer simplificação da estratégia de produção deve ser precedida de novos testes e análise de latência/custo.

## 6. Limitações

- O benchmark contém 14 perguntas e foi construído para a base de conhecimento atual, composta por quatro documentos Markdown. A amostra é pequena e não representa todos os tipos de perguntas que usuários podem fazer.
- As fontes esperadas (`gold_sources`) foram definidas para este benchmark. A qualidade dessas marcações influencia diretamente Precision@4 e Recall@4.
- A avaliação ocorre no nível de documento/fonte, e não no nível de chunk, trecho, resposta ou afirmação factual.
- O benchmark não mede a qualidade da resposta final do agente, a fidelidade da síntese, alucinações, latência ou custo computacional.
- A base documental pequena pode limitar a capacidade de distinguir estratégias: muitas consultas retornam as mesmas fontes relevantes.

## 7. Conclusão

Na avaliação executada em 08/11/2026, a estratégia com expansão de consulta elevou Precision@4 de 0,4643 para 0,4821, Recall@4 de 0,9762 para 1,0000 e MRR de 0,9643 para 1,0000. Os ganhos vieram principalmente das perguntas RAG-02 e RAG-11. A estratégia híbrida empatou com a expandida em todas as métricas agregadas.

A evidência é consistente com a hipótese de que a transformação determinística de consultas ajuda a resolver incompatibilidades de vocabulário, especialmente nos conceitos de receita líquida e Selic. A conclusão deve permanecer delimitada ao conjunto de 14 perguntas e aos quatro documentos avaliados.

## 8. Artefatos reproduzíveis

- Script: `scripts/evaluate_rag_retrieval.py`
- Resultado detalhado: `outputs/rag_retrieval_evaluation.json`
- Resultado tabular: `outputs/rag_retrieval_evaluation.csv`

Comando para reproduzir a avaliação a partir da raiz do repositório:

```powershell
python scripts/evaluate_rag_retrieval.py
```

O script carrega o índice FAISS existente e não executa a etapa de construção do índice.
