import pyodbc

server   = '207.180.243.86'           # deine Public-IP
port     = 1433
database = 'WideWorldImportersDW'     # deine Ziel-DB
username = 'sql-crafter'
password = 'start123'

conn_str = (
    "Driver={ODBC Driver 18 for SQL Server};"
    f"Server={server},{port};"
    f"Database={database};"
    f"UID={username};"
    f"PWD={password};"
    "Encrypt=yes;TrustServerCertificate=yes;"
)

conn = pyodbc.connect(conn_str, timeout=5)
print("✅ Connected as", username)
conn.close()