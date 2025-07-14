import psycopg2

class DatabaseManager:
    def __init__(self, dbname, user, password='mysecretpassword', host='localhost', port='5432'):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cursor = self.conn.cursor()

    def create_table(self, year):
        table_name = f"bitcoin_crawl_{year}"
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                url TEXT,
                title TEXT,
                content TEXT,
                raw_html TEXT,
                publish_date TEXT,
                language TEXT
            )
        """)
        self.conn.commit()

    def save_result(self, year, url, title, content, raw_html, publish_date, language):
        table_name = f"bitcoin_crawl_{year}"
        self.cursor.execute(
            f"""INSERT INTO {table_name} 
                (url, title, content, raw_html, publish_date, language) 
                VALUES (%s, %s, %s, %s, %s, %s)""",
            (url, title, content, raw_html, publish_date, language)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
