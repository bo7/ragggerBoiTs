#!/usr/bin/env python3
import json
import pyodbc

# === CONFIGURE YOUR CONNECTION ===
CONN_STR = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=207.180.243.86,1433;"
    "Database=WideWorldImportersDW;"
    "UID=sql-crafter;"
    "PWD=start123;"
    "Encrypt=yes;TrustServerCertificate=yes;"
)
SAMPLE_LIMIT = 5
OUTPUT_PATH = "all_tables.json"

def get_tables(cursor):
    cursor.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
    """)
    return cursor.fetchall()

def get_columns(cursor, schema, table):
    cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
    """, schema, table)
    return [
        {
            "name": row.COLUMN_NAME,
            "data_type": row.DATA_TYPE,
            "is_nullable": (row.IS_NULLABLE == "YES")
        }
        for row in cursor.fetchall()
    ]

def get_sample_rows(cursor, schema, table, limit):
    # 1) Fetch column names & types
    cursor2 = cursor.connection.cursor()
    cursor2.execute(
        "SELECT COLUMN_NAME, DATA_TYPE "
        "FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
        "ORDER BY ORDINAL_POSITION",
        schema, table
    )
    cols_meta = cursor2.fetchall()

    # 2) Build SELECT clause, casting unsupported types to strings
    select_parts = []
    for col_name, data_type in cols_meta:
        dt = data_type.lower()
        if dt in ("geography", "geometry", "hierarchyid", "sql_variant"):
            select_parts.append(
                f"TRY_CONVERT(NVARCHAR(MAX), [{col_name}]) AS [{col_name}]"
            )
        else:
            select_parts.append(f"[{col_name}]")
    select_clause = ", ".join(select_parts)

    # 3) Execute the sample query
    query = f"SELECT TOP {limit} {select_clause} FROM [{schema}].[{table}]"
    cursor.execute(query)

    cols = [c[0] for c in cursor.description]
    rows = cursor.fetchall()

    # 4) Map each row to a dict
    result = []
    for row in rows:
        result.append({ cols[i]: row[i] for i in range(len(cols)) })
    return result

def main():
    conn = pyodbc.connect(CONN_STR, timeout=10)
    cursor = conn.cursor()

    catalog = {"tables": []}

    for schema, table in get_tables(cursor):
        # skip system/internal schemas if desired
        if schema.upper() in ("SYS", "INFORMATION_SCHEMA"):
            continue

        cols = get_columns(cursor, schema, table)
        samples = get_sample_rows(cursor, schema, table, SAMPLE_LIMIT)

        catalog["tables"].append({
            "schema": schema,
            "name": table,
            "columns": cols,
            "sample_rows": samples
        })
        print(f" • Extracted {len(cols)} columns + {len(samples)} sample rows from {schema}.{table}")

    conn.close()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, default=str)
    print(f"\n✅ Written full catalog + samples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()