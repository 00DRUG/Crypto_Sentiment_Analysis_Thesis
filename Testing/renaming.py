from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()  # this loads variables from .env

conn = psycopg2.connect(
    dbname='postgres',
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT")
)

conn.autocommit = True

cur = conn.cursor()
cur.execute("ALTER DATABASE new_database_name RENAME TO information_db;")

cur.close()
conn.close()

print("✅ Database renamed!")
