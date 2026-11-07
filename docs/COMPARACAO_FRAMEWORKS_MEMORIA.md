# Comparação de Frameworks de Memória para Agentes de IA

**Projeto:** Business Intelligence Agent  
**Data da pesquisa:** 07/11/2026

## 1. Objetivo

Este documento compara três abordagens de memória de longo prazo para agentes de IA: a implementação própria em SQLite utilizada no projeto, LangMem e Mem0.

A comparação considera persistência, recuperação, organização das memórias, integração com agentes, dependências operacionais e adequação ao contexto de Business Intelligence.

A análise dos frameworks externos é baseada em documentação oficial. Não foram executados benchmarks comparativos entre as três soluções.

## 2. Implementação atual: SQLite

O projeto implementa uma camada própria de memória persistente utilizando SQLite.

As memórias são organizadas em três categorias:

- **Semântica:** fatos e conhecimentos que podem ser reutilizados em diferentes interações.
- **Episódica:** registros de experiências e análises realizadas anteriormente pelo agente.
- **Procedural:** regras e orientações sobre como o agente deve executar determinadas tarefas.

O armazenamento é persistente em arquivo local. A recuperação utiliza correspondência lexical de palavras-chave, e as operações de escrita e leitura são controladas pelo código da aplicação.

### Vantagens

- Baixa complexidade operacional.
- Não exige serviço externo de memória.
- Persistência local e controle direto sobre os registros.
- Facilidade de auditoria e inspeção dos dados.
- Baixo custo de infraestrutura.
- Comportamento previsível e adequado a testes determinísticos.

### Limitações

- A busca lexical pode não recuperar informações expressas com sinônimos ou paráfrases.
- A extração de fatos depende das regras implementadas pela aplicação.
- Não oferece, por padrão, consolidação semântica automática de memórias.
- A resolução de informações contraditórias precisa ser tratada pela aplicação.
- A evolução para recuperação vetorial exige implementação adicional.

## 3. LangMem

LangMem é um toolkit desenvolvido pelo ecossistema LangChain para permitir que agentes aprendam e adaptem seu comportamento com base nas interações.

Diferentemente de um banco de dados isolado, LangMem oferece componentes de gerenciamento de memória que podem trabalhar com diferentes sistemas de armazenamento, incluindo a integração com o LangGraph Store.

### Principais recursos

- Extração de informações relevantes de conversas.
- Ferramentas para gerenciar e pesquisar memórias durante a execução do agente (*hot path*).
- Gerenciamento de memórias em segundo plano (*background processing*).
- Organização de informações em estruturas semânticas e coleções.
- Integração com armazenamento persistente do LangGraph.
- Atualização de instruções e otimização de prompts.

### Funcionamento

O LangMem oferece duas estratégias principais:

**Hot path:** o próprio fluxo do agente utiliza ferramentas de memória para pesquisar e registrar informações durante a interação.

**Background:** um processo separado analisa as interações, extrai informações relevantes e consolida o conhecimento sem necessariamente aumentar a latência da resposta principal.

A biblioteca oferece componentes como:

- `create_manage_memory_tool`
- `create_search_memory_tool`
- `create_memory_manager`
- `create_memory_store_manager`

### Vantagens

- Integração natural com LangChain e LangGraph.
- Flexibilidade na escolha do armazenamento.
- Suporte a extração e atualização de memórias estruturadas.
- Separação entre memória operacional e processamento em segundo plano.
- Adequação a agentes que precisam evoluir suas instruções e conhecimento.

### Limitações

- A extração e consolidação inteligente dependem de um modelo de linguagem.
- A arquitetura exige configuração adequada do armazenamento.
- O uso de armazenamento em memória, como `InMemoryStore`, não oferece persistência após reinicialização.
- A integração completa exige maior configuração do que uma solução SQLite simples.

### Adequação ao Business Intelligence Agent

LangMem seria uma evolução natural para o projeto, pois o agente já utiliza LangGraph.

Poderia substituir parte da lógica manual de gerenciamento de memória por ferramentas especializadas para identificar, registrar e recuperar decisões comerciais, preferências e experiências anteriores.

A integração também permitiria separar o registro de informações relevantes da execução principal das consultas de BI.

## 4. Mem0

Mem0 é uma solução especializada em memória para aplicações de IA. Disponibiliza uma biblioteca open source e uma plataforma gerenciada.

Seu objetivo é armazenar informações relevantes extraídas de interações, evitando a necessidade de inserir todo o histórico de conversas novamente no contexto do modelo.

### Principais recursos

- Extração automática de fatos relevantes.
- Deduplicação de informações.
- Recuperação por similaridade semântica.
- Busca lexical e por entidades.
- Organização por identificadores de usuário, agente, aplicação e sessão.
- Operações de atualização e exclusão.
- Armazenamento vetorial e metadados.
- Recursos de histórico e ciclo de vida das memórias.

### Funcionamento

O fluxo básico possui duas etapas:

**1. Escrita:** a aplicação envia mensagens para `add()`. O sistema identifica fatos relevantes, processa as informações e armazena as memórias.

**2. Recuperação:** antes de uma nova resposta, a aplicação executa `search()` para recuperar informações relevantes e adicioná-las ao contexto do modelo.

A documentação descreve diferentes sinais de recuperação:

- **Semântico:** similaridade vetorial entre conceitos.
- **Lexical:** correspondência de palavras e expressões.
- **Entidades:** associação entre pessoas, organizações e conceitos.
- **Temporal:** relevância considerando informações de tempo e recência.

### Vantagens

- Especialização em gerenciamento de memória para agentes.
- Recuperação híbrida.
- Extração automatizada de informações relevantes.
- Deduplicação de fatos.
- Escopo de memória separado por usuário, agente, aplicação e sessão.
- Operações explícitas de atualização e exclusão.
- Alternativa open source para implantação própria.

### Limitações

- Maior quantidade de componentes operacionais do que uma solução SQLite simples.
- A extração automática exige um modelo de linguagem.
- A recuperação vetorial exige configuração de embeddings e armazenamento vetorial.
- A qualidade das memórias depende da configuração dos modelos e mecanismos de recuperação.
- A plataforma gerenciada introduz dependência de serviço externo e credenciais.

### Adequação ao Business Intelligence Agent

Mem0 seria interessante caso o agente evoluísse para atender vários usuários ou equipes comerciais, mantendo memórias individualizadas.

Por exemplo, poderia armazenar decisões anteriores, preferências de análise, indicadores frequentemente consultados e contexto específico de cada usuário.

A recuperação híbrida também permitiria localizar uma decisão mesmo quando a pergunta utilizasse palavras diferentes das registradas originalmente.

## 5. Comparação técnica

| Critério | SQLite atual | LangMem | Mem0 |
|---|---|---|---|
| Categoria | Implementação própria | Toolkit de memória | Plataforma e biblioteca |
| Persistência | SQLite local | LangGraph Store ou outro backend | Armazenamentos configuráveis |
| Busca lexical | Sim | Depende do Store | Sim |
| Busca semântica | Não | Com índice e embeddings configurados | Sim |
| Extração automática | Não, depende do código | Sim, com modelo de linguagem | Sim, com modelo de linguagem |
| Consolidação | Manual | Gerenciador de memória | Processamento de memória |
| Integração LangGraph | Implementação própria | Nativa | Integração pela aplicação |
| Memória por usuário | Exige implementação | Namespaces | Identificadores próprios |
| Atualização e exclusão | SQL próprio | Gerenciadores e Store | Operações específicas |
| Complexidade inicial | Baixa | Média | Média a alta |

## 6. Aplicação dos tipos de memória

As três abordagens podem contribuir para diferentes tipos de memória utilizados em agentes.

| Tipo | Exemplo no projeto | Possível aplicação |
|---|---|---|
| Semântica | Definições de indicadores e fatos comerciais | Guardar conhecimento reutilizável |
| Episódica | Histórico de análises de vendas | Recuperar experiências anteriores |
| Procedural | Orientações de execução e regras de negócio | Melhorar consistência das tarefas |

A memória procedural merece atenção especial: ela pode ser representada por instruções do agente, prompts e regras da aplicação, não somente por registros convencionais de texto.

## 7. Comparação de adequação ao projeto

### SQLite

É a alternativa mais adequada para o estágio atual do projeto.

Permite demonstrar persistência, separação dos tipos de memória e recuperação entre sessões sem adicionar serviços externos ou dependências de modelos pagos.

Sua principal limitação é a recuperação lexical, que pode deixar de encontrar memórias relevantes quando a consulta apresenta uma formulação diferente.

### LangMem

É a alternativa mais alinhada à arquitetura existente.

Como o agente já utiliza LangGraph, a integração com Store, ferramentas de gerenciamento e processamento em segundo plano pode reduzir código próprio e permitir evolução da memória sem alterar toda a arquitetura.

### Mem0

É uma alternativa interessante para uma aplicação de memória mais independente do framework de agentes.

Sua recuperação híbrida, identificação de entidades e escopo por usuário podem beneficiar um ambiente de BI com múltiplos usuários e histórico de interações mais extenso.

## 8. Recomendação técnica

Para a arquitetura atual do Business Intelligence Agent, recomenda-se manter SQLite como mecanismo implementado e utilizar LangMem como principal referência para uma evolução integrada ao LangGraph.

Mem0 deve ser considerado como alternativa quando a necessidade de gerenciamento de memória entre diferentes usuários, agentes e aplicações justificar a adoção de uma camada especializada.

A decisão final depende de requisitos de escala, qualidade da recuperação, custo operacional, privacidade e necessidade de automação na extração de informações.

Não foi realizada uma comparação quantitativa de latência, precisão de recuperação, custo ou qualidade de respostas entre os três mecanismos. As conclusões representam uma análise arquitetural e funcional baseada nas capacidades documentadas.

## 9. Fontes oficiais

### LangMem

- Introdução e funcionalidades: https://langchain-ai.github.io/langmem/
- Referência de memória: https://langchain-ai.github.io/langmem/reference/memory/
- Conceitos de memória do LangGraph: https://docs.langchain.com/oss/python/concepts/memory

### Mem0

- Como funciona: https://github.com/mem0ai/mem0/blob/main/docs/core-concepts/how-it-works.mdx
- Visão geral open source: https://github.com/mem0ai/mem0/blob/main/docs/open-source/overview.mdx
- Configuração de componentes: https://github.com/mem0ai/mem0/blob/main/docs/open-source/configuration.mdx
- Referência de API: https://github.com/mem0ai/mem0/blob/main/skills/mem0/references/api-reference.md

**Consulta às fontes:** 07/11/2026.

## Implementacao atual: recuperacao hibrida de memoria

A implementacao do projeto evoluiu de uma busca exclusivamente lexical
para uma estrategia hibrida, combinando SQLite e FAISS.

### Armazenamento

O SQLite continua sendo a fonte de verdade das memorias persistidas.
Os registros sao organizados em tres categorias:

- Semanticas: conceitos e conhecimentos reutilizaveis.
- Episodicas: historico de interacoes e eventos.
- Procedurais: instrucoes e procedimentos.

### Recuperacao lexical

A busca lexical normaliza os termos da consulta, remove acentos,
considera palavras completas e aplica uma lista de stopwords.
Isso reduz correspondencias acidentais por substrings.

### Recuperacao semantica

A busca semantica utiliza embeddings multilingues do modelo
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
com um indice vetorial FAISS.

O indice e derivado dos registros armazenados no SQLite.
Um fingerprint permite identificar alteracoes nas memorias e
reconstruir o indice quando necessario.

### Combinacao dos resultados

A recuperacao hibrida combina os rankings lexical e semantico por
Reciprocal Rank Fusion (RRF).

O valor RRF representa uma pontuacao de combinacao de rankings;
nao deve ser interpretado como probabilidade ou nivel de confianca.

### Validacao

Os testes automatizados verificam recuperacao por sinonimos e
parafraseamentos, combinacao lexical e semantica, filtros por
categoria, atualizacao do indice e comportamento com armazenamento
vazio.

Essa arquitetura preserva a simplicidade e persistencia do SQLite,
enquanto amplia a capacidade de recuperar memorias expressas com
vocabulario diferente daquele utilizado no momento do armazenamento.

