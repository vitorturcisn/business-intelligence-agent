# Business Intelligence Agent

Projeto de pÃ³s-graduaÃ§Ã£o em agentes de IA, desenvolvido com Python, LangChain e LangGraph. A soluÃ§Ã£o combina consultas SQL, recuperaÃ§Ã£o documental (RAG), consulta a uma API econÃ´mica, memÃ³ria conversacional e persistente e verificaÃ§Ã£o factual baseada em Natural Language Inference (NLI).

## 1. Objetivo

Construir um agente de Business Intelligence capaz de responder perguntas sobre dados de vendas e regras de negÃ³cio, consultar uma fonte externa de indicadores econÃ´micos e preservar informaÃ§Ãµes relevantes entre interaÃ§Ãµes.

O projeto explora chains, agentes e ferramentas, diferentes estratÃ©gias de orquestraÃ§Ã£o, RAG, memÃ³ria de curto e longo prazo, verificaÃ§Ã£o de respostas e observabilidade.

## 2. Arquitetura

```mermaid
flowchart TD
    U[Usuario] --> O[Orquestrador]
    O -->|Pergunta quantitativa mensal| SQL[Ferramenta SQL]
    O -->|Pergunta conceitual| RAG[Recuperacao RAG]
    O -->|Recordacao explicita| PM[Memoria pessoal]
    O -->|Pergunta geral| HM[Busca hibrida de memoria]
    O -->|Consulta economica| BCB[API Banco Central]

    SQL --> DB[(SQLite - vendas)]
    RAG --> KB[Base documental]
    KB --> FAISS[(FAISS - conhecimento)]
    RAG --> Q[Qwen3 + AgentExecutor]
    HM --> MDB[(SQLite - memorias)]
    HM --> VDB[(FAISS - indice semantico)]
    PM --> MDB
    BCB --> SGS[Serie SGS 432 - Selic]

    Q --> V[Verificacao de resposta]
    SQL --> V
    PM --> V
    HM --> V
    BCB --> V
    V --> NLI[Verificador NLI + guardrails]
    NLI --> RESP[Resposta liberada ou bloqueada]
    Q -. rastreamento .-> LS[LangSmith]
```

O repositÃ³rio reÃºne mais de uma abordagem de agentes para fins didÃ¡ticos. A demonstraÃ§Ã£o interativa integrada `scripts/run_local_openai_tools_agent.py`, validada com o modelo local Qwen3, utiliza `create_openai_tools_agent` e `AgentExecutor`. HÃ¡ tambÃ©m demonstraÃ§Ãµes de outras estratÃ©gias, incluindo `create_agent` e ReAct clÃ¡ssico.

Na demonstraÃ§Ã£o integrada, o orquestrador aplica rotas determinÃ­sticas antes de recorrer ao modelo:

1. Perguntas de recordaÃ§Ã£o pessoal consultam diretamente memÃ³rias explicitamente salvas.
2. Perguntas quantitativas mensais sÃ£o resolvidas por SQL, que permanece como fonte de verdade para os valores.
3. Perguntas conceituais consultam primeiro a base RAG; as evidÃªncias recuperadas sÃ£o entÃ£o fornecidas ao Qwen3 para formular a resposta.
4. Perguntas gerais podem recuperar contexto por meio da busca hÃ­brida de memÃ³ria antes de chamar o agente.

As respostas geradas sÃ£o verificadas contra as observaÃ§Ãµes das ferramentas. A resposta original sÃ³ Ã© liberada quando o verificador classifica todas as afirmaÃ§Ãµes como sustentadas.

## 3. Componentes

### Chains e processamento

- `PromptTemplate` e `ChatPromptTemplate`.
- LCEL e composiÃ§Ã£o de componentes.
- SaÃ­da estruturada com Pydantic.
- ExecuÃ§Ã£o `invoke`, `batch` e `stream`.
- Chains sequenciais e paralelas e roteamento.
- Few-shot prompting e estrutura PTCF.

### Agentes e ferramentas

- DemonstraÃ§Ã£o integrada com `create_openai_tools_agent` e `AgentExecutor`.
- ExecuÃ§Ã£o local do Qwen3 por meio da interface compatÃ­vel com a API da OpenAI fornecida pelo Ollama.
- DemonstraÃ§Ãµes adicionais com `create_agent` e ReAct clÃ¡ssico.
- Ferramenta SQL para consultas somente leitura Ã  base SQLite de vendas.
- Ferramenta RAG para recuperaÃ§Ã£o de conhecimento documental.
- Ferramenta conectada Ã  API SGS do Banco Central do Brasil, sÃ©rie 432 (Selic).
- Ferramentas para gravar e recuperar memÃ³rias.

### RAG

A base documental estÃ¡ em `data/knowledge_base/` e contÃ©m glossÃ¡rio de KPIs, dicionÃ¡rio de dados, regras de negÃ³cio e informaÃ§Ãµes sobre fontes e limitaÃ§Ãµes.

A recuperaÃ§Ã£o utiliza embeddings multilÃ­ngues e FAISS. O projeto inclui expansÃ£o de consultas e recuperaÃ§Ã£o de trechos relevantes. Na verificaÃ§Ã£o, os trechos recuperados sÃ£o separados por documento de origem para que cada afirmaÃ§Ã£o seja comparada com evidÃªncias identificÃ¡veis.

### MemÃ³ria

A soluÃ§Ã£o contempla memÃ³ria de curto prazo e memÃ³ria de longo prazo:

- Na demonstraÃ§Ã£o integrada com Qwen3, o histÃ³rico conversacional mantÃ©m atÃ© oito mensagens anteriores em memÃ³ria RAM e Ã© perdido quando o processo termina.
- A memÃ³ria de longo prazo Ã© persistida em SQLite e organizada em categorias semÃ¢ntica, episÃ³dica e procedural.
- A recuperaÃ§Ã£o de longo prazo combina busca lexical normalizada e busca semÃ¢ntica com embeddings e FAISS.
- Os rankings sÃ£o combinados por Reciprocal Rank Fusion (RRF). O SQLite Ã© a fonte de verdade; o Ã­ndice FAISS Ã© derivado e reconstruÃ­vel.
- MemÃ³rias pessoais explÃ­citas sÃ£o gravadas quando o usuÃ¡rio utiliza uma solicitaÃ§Ã£o de salvamento, como â€œLembre-se que...â€ ou â€œGuarde na memÃ³ria:...â€.

O repositÃ³rio tambÃ©m contÃ©m demonstraÃ§Ãµes de estratÃ©gias de persistÃªncia de estado conversacional com LangGraph.

### VerificaÃ§Ã£o factual

Foi implementado um verificador local de Natural Language Inference (NLI), baseado no modelo multilÃ­ngue `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, executado com Transformers e PyTorch.

O fluxo:

- extrai afirmaÃ§Ãµes da resposta;
- compara cada afirmaÃ§Ã£o com evidÃªncias provenientes das ferramentas;
- classifica a relaÃ§Ã£o como `APOIA`, `REFUTA` ou `NEUTRA`;
- aplica verificaÃ§Ãµes auxiliares de consistÃªncia numÃ©rica e coerÃªncia lÃ³gica;
- produz um veredito geral: `SUSTENTADA`, `REFUTADA`, `INCONCLUSIVA` ou `INCOERENTE`.

Quando a resposta nÃ£o Ã© sustentada, ela nÃ£o Ã© liberada pelo executor. O verificador Ã© uma camada de apoio Ã  validaÃ§Ã£o, nÃ£o uma garantia absoluta de correÃ§Ã£o factual. Os escores do classificador nÃ£o devem ser interpretados automaticamente como probabilidades calibradas.

### Observabilidade e avaliaÃ§Ã£o

O projeto integra LangSmith para rastreamento e avaliaÃ§Ã£o. O conjunto de avaliaÃ§Ã£o contÃ©m cinco cenÃ¡rios e duas mÃ©tricas: cobertura de termos esperados na resposta (*answer coverage*) e aderÃªncia Ã  trajetÃ³ria/ferramentas esperadas (*trajectory*).

**LimitaÃ§Ã£o importante:** a avaliaÃ§Ã£o automatizada do agente usa um modelo determinÃ­stico simulado. Os resultados medem o comportamento do fluxo controlado de teste, nÃ£o a qualidade geral ou autonomia de um LLM em produÃ§Ã£o. A demonstraÃ§Ã£o integrada com Qwen3 Ã© executada separadamente, em ambiente local.

## 4. Tecnologias

- Python 3.12
- LangChain, LangGraph e LangSmith
- Ollama e Qwen3 (`qwen3:1.7b`) para a demonstraÃ§Ã£o integrada local
- SQLite
- FAISS
- Hugging Face Transformers e Sentence Transformers
- PyTorch
- Pydantic
- Pandas e Requests
- Pytest

## 5. Estrutura do projeto

```text
business-intelligence-agent/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ knowledge_base/
â”‚   â””â”€â”€ sample/
â”œâ”€â”€ docs/
â”œâ”€â”€ outputs/
â”œâ”€â”€ scripts/
â”œâ”€â”€ src/business_agent/
â”‚   â”œâ”€â”€ agents/
â”‚   â”œâ”€â”€ chains/
â”‚   â”œâ”€â”€ memory/
â”‚   â”œâ”€â”€ observability/
â”‚   â”œâ”€â”€ retrieval/
â”‚   â”œâ”€â”€ tools/
â”‚   â””â”€â”€ verification/
â”œâ”€â”€ tests/
â”œâ”€â”€ .env.example
â”œâ”€â”€ .gitignore
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

## 6. ConfiguraÃ§Ã£o do ambiente

Requisitos: Python 3.12, Git, Ollama (para a demonstraÃ§Ã£o integrada com Qwen3) e acesso Ã  internet para instalar dependÃªncias e baixar os modelos do Hugging Face na primeira execuÃ§Ã£o.

No PowerShell, na raiz do repositÃ³rio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Crie o arquivo local de variÃ¡veis de ambiente:

```powershell
Copy-Item .env.example .env
```

Para habilitar o rastreamento remoto, configure `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` e `LANGSMITH_PROJECT` no `.env`. NÃ£o versione nem compartilhe o arquivo `.env` ou suas chaves.

O script `run_local_openai_tools_agent.py` configura diretamente o endpoint local `http://localhost:11434/v1`, o modelo `qwen3:1.7b` e a chave local de compatibilidade `ollama`. Portanto, essa demonstraÃ§Ã£o nÃ£o precisa de crÃ©ditos nem de uma chave da API OpenAI. As variÃ¡veis `OPENAI_API_KEY` e `OPENAI_MODEL` presentes no `.env.example` sÃ£o usadas por outras demonstraÃ§Ãµes/configuraÃ§Ãµes, nÃ£o pela execuÃ§Ã£o local descrita nesta seÃ§Ã£o.

### Preparar o Qwen3 no Ollama

Com o Ollama instalado e em execuÃ§Ã£o, baixe o modelo utilizado pelo script:

```powershell
ollama pull qwen3:1.7b
```

O script espera que o serviÃ§o local do Ollama esteja acessÃ­vel em `http://localhost:11434`. Se alterar o modelo ou endpoint, ajuste a configuraÃ§Ã£o em `scripts/run_local_openai_tools_agent.py`.

## 7. PreparaÃ§Ã£o dos dados e Ã­ndices

A base SQLite de vendas de demonstraÃ§Ã£o estÃ¡ em `data/sample/sales.db` e tambÃ©m estÃ¡ versionada no repositÃ³rio. Para recriÃ¡-la:

```powershell
python scripts/setup_data.py
```

**AtenÃ§Ã£o:** esse script recria a base de demonstraÃ§Ã£o; alteraÃ§Ãµes locais em `sales.db` podem ser substituÃ­das.

Para construir o Ã­ndice vetorial da base documental:

```powershell
python scripts/build_faiss_index.py
```

Na primeira execuÃ§Ã£o, os modelos de embeddings e NLI podem precisar ser baixados do Hugging Face. A inicializaÃ§Ã£o pode levar mais tempo na primeira execuÃ§Ã£o.

## 8. Executar demonstraÃ§Ãµes

### Agente integrado local: Qwen3 + SQL + RAG + memÃ³ria + NLI

Com o ambiente virtual ativado e o Ollama em execuÃ§Ã£o:

```powershell
python scripts/run_local_openai_tools_agent.py
```

O programa disponibiliza as ferramentas `sales_monthly_summary`, `business_knowledge_search`, `get_selic_target` e `search_business_memory`. Digite perguntas em portuguÃªs no terminal; digite `sair` para encerrar.

Exemplos de perguntas para demonstrar as rotas:

```text
Qual foi a receita lÃ­quida em setembro de 2026?
Explique a diferenÃ§a entre receita bruta e receita lÃ­quida.
Qual Ã© a meta Selic mais recente?
Lembre-se que nosso KPI principal Ã© receita lÃ­quida.
Qual KPI eu mencionei como principal?
```

As perguntas quantitativas mensais sÃ£o resolvidas pela rota SQL determinÃ­stica. As perguntas conceituais consultam o RAG antes da geraÃ§Ã£o com Qwen3. A recuperaÃ§Ã£o de memÃ³rias pessoais explÃ­citas Ã© feita diretamente na memÃ³ria persistente. As respostas passam pelo verificador antes de serem liberadas.

### Chains

```powershell
python scripts/run_initial_chain.py --mode simple
python scripts/run_initial_chain.py --mode batch
python scripts/run_initial_chain.py --mode stream
```

Os modos demonstrativos utilizam implementaÃ§Ãµes simuladas/determinÃ­sticas; nÃ£o devem ser apresentados como inferÃªncia realizada por um LLM externo.

### Outras demonstraÃ§Ãµes de agentes

```powershell
python scripts/run_agent_mock.py
python scripts/run_react_demo.py
python scripts/run_legacy_openai_tools_demo.py
```

As demonstraÃ§Ãµes de agentes com mock exercitam o fluxo de chamadas de ferramentas sem depender de crÃ©ditos de LLM.

### RAG e memÃ³ria

```powershell
python scripts/test_rag_search.py
python scripts/run_memory_demo.py
python scripts/run_persistent_agent_demo.py
python scripts/run_hybrid_memory_demo.py
```

### Verificador NLI

```powershell
python scripts/run_verifier_demo.py
```

### IntegraÃ§Ã£o com Banco Central

```powershell
python scripts/check_bcb_api.py
```

Esse teste consulta a API externa e depende de conectividade.

## 9. LangSmith

Verificar a configuraÃ§Ã£o:

```powershell
python scripts/check_langsmith.py
```

Executar um rastreamento:

```powershell
python scripts/run_langsmith_trace.py
```

Executar a avaliaÃ§Ã£o local:

```powershell
python scripts/evaluate_business_agent.py
```

Enviar o conjunto de avaliaÃ§Ã£o e executar a avaliaÃ§Ã£o remota:

```powershell
python scripts/evaluate_business_agent.py --upload
```

O resultado local da avaliaÃ§Ã£o Ã© salvo em `outputs/business_agent_evaluation.json`.

## 10. Testes automatizados

Executar a suÃ­te completa:

```powershell
python -m pytest -q
```

Na Ãºltima validaÃ§Ã£o registrada durante a preparaÃ§Ã£o desta entrega, **112 testes passaram**. A suÃ­te pode mudar conforme novas funcionalidades e testes forem adicionados. Os avisos de depreciaÃ§Ã£o emitidos por algumas dependÃªncias nÃ£o representaram falhas nessa execuÃ§Ã£o.

## 11. DocumentaÃ§Ã£o complementar

A comparaÃ§Ã£o entre frameworks de memÃ³ria estÃ¡ em `docs/COMPARACAO_FRAMEWORKS_MEMORIA.md`. O documento compara SQLite, LangMem e Mem0 e registra os critÃ©rios e decisÃµes arquiteturais adotados.

## 12. LimitaÃ§Ãµes e consideraÃ§Ãµes

- Os modelos determinÃ­sticos simulados permitem reproduzir fluxos sem depender de crÃ©ditos de LLM; nÃ£o demonstram autonomia ou qualidade de um modelo generativo real.
- A avaliaÃ§Ã£o no LangSmith utiliza cinco cenÃ¡rios controlados e nÃ£o representa uma validaÃ§Ã£o abrangente da qualidade de respostas de LLM.
- A demonstraÃ§Ã£o integrada usa Qwen3 local por meio do Ollama; seu comportamento depende do modelo e dos recursos disponÃ­veis na mÃ¡quina.
- O histÃ³rico conversacional da demonstraÃ§Ã£o integrada fica em RAM e Ã© perdido ao encerrar o processo; a memÃ³ria de longo prazo Ã© persistida separadamente.
- RRF combina posiÃ§Ãµes nos rankings e nÃ£o Ã© uma probabilidade de relevÃ¢ncia.
- Os escores NLI nÃ£o sÃ£o necessariamente probabilidades calibradas.
- A verificaÃ§Ã£o numÃ©rica Ã© auxiliar e nÃ£o substitui uma validaÃ§Ã£o semÃ¢ntica completa.
- Algumas demonstraÃ§Ãµes preservam componentes clÃ¡ssicos do LangChain para fins didÃ¡ticos e podem emitir avisos de depreciaÃ§Ã£o.

## 13. Status

A implementaÃ§Ã£o contempla chains, agentes, ferramentas, RAG, memÃ³ria, verificaÃ§Ã£o e observabilidade. A Ãºltima execuÃ§Ã£o da suÃ­te automatizada registrou 112 testes aprovados. A demonstraÃ§Ã£o integrada local tambÃ©m foi validada com uma pergunta conceitual sobre receita bruta e lÃ­quida, com as afirmaÃ§Ãµes apoiadas pelas evidÃªncias recuperadas e veredito geral `SUSTENTADA`.

A revisÃ£o final dos requisitos acadÃªmicos, a consolidaÃ§Ã£o do relatÃ³rio e a preparaÃ§Ã£o da apresentaÃ§Ã£o sÃ£o etapas de finalizaÃ§Ã£o da entrega.

