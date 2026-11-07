# Dicionário de dados — Banco de vendas

O projeto utiliza um banco SQLite local chamado sales.db.
A tabela principal é sales.

## Colunas

- sale_id: identificador do pedido.
- sale_date: data da venda.
- product: nome do produto vendido.
- category: categoria do produto.
- region: região associada à venda.
- quantity: quantidade de unidades vendidas.
- unit_price: preço unitário do produto.
- discount_pct: percentual de desconto aplicado à venda.

## Cálculos derivados

Receita bruta por item:
quantity × unit_price

Valor do desconto por item:
quantity × unit_price × discount_pct

Receita líquida por item:
quantity × unit_price × (1 - discount_pct)

## Cuidados

O número de pedidos deve considerar sale_id distintos.
A quantidade de unidades deve ser calculada pela soma de quantity.
A receita deve considerar os descontos para representar a receita líquida.
