"""Cria a base SQLite de demonstração do agente."""

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "sample"
DB_PATH = DATA_DIR / "sales.db"


SALES_DATA = [
    ("A001", "2026-08-03", "Laptop", "Eletrônicos", "Sul", 2, 3500.00, 0.05),
    ("A002", "2026-08-05", "Mouse", "Acessórios", "Sudeste", 10, 80.00, 0.10),
    ("A003", "2026-08-10", "Cadeira", "Móveis", "Sul", 4, 450.00, 0.00),
    ("A004", "2026-08-15", "Celular", "Eletrônicos", "Sudeste", 3, 2200.00, 0.05),
    ("A005", "2026-08-20", "Teclado", "Acessórios", "Nordeste", 8, 150.00, 0.05),
    ("A006", "2026-08-25", "Mesa", "Móveis", "Sudeste", 2, 800.00, 0.10),
    ("S001", "2026-09-02", "Laptop", "Eletrônicos", "Sul", 3, 3500.00, 0.05),
    ("S002", "2026-09-06", "Mouse", "Acessórios", "Sudeste", 15, 80.00, 0.10),
    ("S003", "2026-09-09", "Cadeira", "Móveis", "Sul", 5, 450.00, 0.00),
    ("S004", "2026-09-13", "Celular", "Eletrônicos", "Sudeste", 4, 2200.00, 0.05),
    ("S005", "2026-09-18", "Teclado", "Acessórios", "Nordeste", 10, 150.00, 0.05),
    ("S006", "2026-09-24", "Mesa", "Móveis", "Sudeste", 3, 800.00, 0.10),
]


def create_database() -> None:
    """Cria a tabela e insere novamente os dados de demonstração."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DROP TABLE IF EXISTS sales")

        connection.execute(
            """
            CREATE TABLE sales (
                sale_id TEXT PRIMARY KEY,
                sale_date TEXT NOT NULL,
                product TEXT NOT NULL,
                category TEXT NOT NULL,
                region TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                unit_price REAL NOT NULL CHECK (unit_price >= 0),
                discount_pct REAL NOT NULL
                    CHECK (discount_pct >= 0 AND discount_pct <= 1)
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO sales (
                sale_id,
                sale_date,
                product,
                category,
                region,
                quantity,
                unit_price,
                discount_pct
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            SALES_DATA,
        )

        connection.commit()

    print("Base de vendas criada com sucesso.")
    print(f"Arquivo: {DB_PATH}")
    print(f"Registros inseridos: {len(SALES_DATA)}")


if __name__ == "__main__":
    create_database()