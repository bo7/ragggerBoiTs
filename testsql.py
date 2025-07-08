import pyodbc
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

server   = os.getenv('DB_SERVER')
port     = os.getenv('DB_PORT')
database = os.getenv('DB_DATABASE')
username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')

conn_str = (
    f"Driver={{{os.getenv('DB_DRIVER')}}};"
    f"Server={server},{port};"
    f"Database={database};"
    f"UID={username};"
    f"PWD={password};"
    "Encrypt=yes;TrustServerCertificate=yes;"
)

conn = pyodbc.connect(conn_str, timeout=5)
print("✅ Connected as", username)
conn.close()