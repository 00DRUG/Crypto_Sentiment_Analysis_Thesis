import sqlite3
from datetime import datetime, timedelta, timezone

DB_NAME = "crypto_predictions.db"


def create_predictions():
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


def delete_old_predictions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    threshold_date = datetime.now(timezone.utc) - timedelta(days=30)
    cursor.execute("DELETE FROM predictions WHERE timestamp < ?", (threshold_date,))

    conn.commit()
    conn.close()


"""
def delete_irrelevant_predictions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT id, text FROM predictions")
    predictions = cursor.fetchall()

    for pred_id, text in predictions:
        # Process the text with spaCy
        doc = nlp(text)

        # Detect future tense by checking if there's a future verb or certain keywords
        future_detected = False

        # Check for modal verbs or future tense keywords
        for token in doc:
            if token.tag_ in ["MD", "VB", "VBP", "VBZ"] and token.text.lower() in ["will", "shall", "expect",
                                                                                   "predict"]:
                future_detected = True
                break  # Once a future prediction is detected, no need to continue checking

        # If no future tense is detected, delete the prediction
        if not future_detected:
            print(
                f"Deleting irrelevant prediction: {text[:30]}...") 
            cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))

    conn.commit()
    conn.close()
"""
