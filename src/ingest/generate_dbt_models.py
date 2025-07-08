import json, pyodbc
from pathlib import Path

# ———————————————————————————
# CONFIGURATION
CONN_STR = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=207.180.243.86,1433;"
    "Database=WideWorldImportersDW;"
    "UID=sql-crafter;"
    "PWD=start123;"
    "Encrypt=yes;TrustServerCertificate=yes;"
)
INPUT_JSON = Path(__file__).parent / "all_tables.json"
MODELS_DIR = Path(__file__).parents[1] / "models"
SOURCES_YML = MODELS_DIR / "sources.yml"
# ———————————————————————————

def main():
    data = json.loads(INPUT_JSON.read_text())
    # 1) Build sources.yml
    sources = {}
    for tbl in data["tables"]:
        schema, name = tbl["schema"], tbl["name"]
        sources.setdefault(schema, []).append(name)
    SOURCES_YML.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_YML, "w") as f:
        f.write("version: 2\n\nsources:\n")
        for schema, names in sources.items():
            f.write(f"  - name: {schema.lower()}\n")
            f.write(f"    database: WideWorldImportersDW\n")
            f.write(f"    schema: {schema}\n")
            f.write("    tables:\n")
            for n in names:
                f.write(f"      - name: {n}\n")
            f.write("\n")

    # 2) Generate one view‐model per table
    for tbl in data["tables"]:
        schema, name = tbl["schema"].lower(), tbl["name"]
        out = MODELS_DIR / schema / f"{name.lower()}.sql"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "{{ config(materialized='view') }}\n"
            f"select * from {{{{ source('{schema}', '{name}') }}}}\n"
        )
        print("✓", schema, name)

    print("\ Scaffold complete — run `dbt run` next.")

# ———————————————————————————
# Ensure main() is called when script is executed
if __name__ == '__main__':
    main()