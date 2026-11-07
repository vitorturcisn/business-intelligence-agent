# Fontes de dados e limitações do projeto

## Dados de vendas
Fonte: banco SQLite local data/sample/sales.db.
Tabela utilizada: sales.
Os dados são sintéticos e foram criados para demonstrar as funcionalidades do agente.

## Indicador Selic
Fonte externa: API de dados do Banco Central do Brasil (SGS).
O projeto consulta a série 432 por meio da ferramenta get_selic_target.
A consulta utiliza datas de início e fim e retorna observações do período solicitado.

## Limitações e cuidados
Os dados comerciais são fictícios e não representam uma empresa real.
A disponibilidade da API do Banco Central depende de conexão externa.
Se uma fonte estiver indisponível, o agente deve informar que não conseguiu consultá-la.
O agente não deve inventar valores para preencher dados ausentes.
Respostas sobre vendas devem ser fundamentadas no banco SQLite.
Respostas sobre regras e definições devem consultar os documentos desta base de conhecimento.
Respostas sobre a Selic devem utilizar os dados retornados pela API.
