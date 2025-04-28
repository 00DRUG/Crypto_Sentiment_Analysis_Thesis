import psycopg2

conn = psycopg2.connect(
    dbname='postgres',         # ← connect to default db (not your 'mydatabase' yet)
    user='postgres',           # ← username
    password='mysecretpassword',  # ← password you set in docker run
    host='localhost',          # ← where postgres is running
    port='5432'                # ← default port
)

conn.autocommit = True

cur = conn.cursor()
cur.execute("ALTER DATABASE mydatabase RENAME TO new_database_name;")

cur.close()
conn.close()

print("✅ Database renamed!")
