# Regras de negócio e orientações de análise

## Períodos
As análises mensais devem utilizar o formato YYYY-MM.
Exemplo: 2026-09 representa setembro de 2026.

## Receita
Sempre diferenciar receita bruta de receita líquida.
Quando a pergunta não especificar o tipo de receita, informar claramente qual indicador foi utilizado.

## Pedidos e unidades
Pedidos representam identificadores distintos de sale_id.
Unidades representam a soma da coluna quantity.
Esses indicadores não são intercambiáveis.

## Descontos
O percentual de desconto deve ser aplicado sobre o valor bruto do item.
A receita líquida corresponde ao valor bruto menos o desconto.

## Comparações
Ao comparar períodos, apresentar os valores utilizados e a variação calculada.
Não afirmar crescimento ou queda sem comparar períodos equivalentes.

## Limitações dos dados
O banco sales.db contém dados de exemplo criados para este projeto.
Não tratar os registros como dados reais de uma empresa.
Não inventar períodos, produtos, categorias ou regiões que não estejam presentes no banco.
Quando não houver dados suficientes, informar a limitação.
