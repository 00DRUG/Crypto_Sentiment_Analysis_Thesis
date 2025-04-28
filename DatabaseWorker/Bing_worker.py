import sqlite3
from datetime import datetime, timedelta, timezone

DB_NAME = "crypto_predictions.db"


def create_table_if_not_exist():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL UNIQUE,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            website_id INTEGER
        )
    ''')
    conn.commit()
    conn.close()


def save_predictions(predictions, website_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for pred in predictions:
        # Check if the prediction text already exists in the database
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE text = ?", (pred,))
        count = cursor.fetchone()[0]

        if count == 0:  # If no match is found, insert the prediction
            cursor.execute("INSERT INTO predictions (text, website_id) VALUES (?, ?)", (pred, website_id))

    conn.commit()
    conn.close()