import sqlite3
import requests
import spacy
import json
import random
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone

from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

DB_NAME = "crypto_predictions.db"

json_file_path = 'links.json'
user_agents_file_path = 'user_agents.json'


# Function to load user agents from a JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Function to create the SQLite database and table if they don't exist
def create_db():
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


# Function to scrape data from a given URL with user-agent rotation
def scrape_data(url, user_agents, data):
    for attempt in range(3):
        browser_category = random.choice(list(user_agents.keys()))
        user_agent = random.choice(user_agents[browser_category])
        headers = {'User-Agent': user_agent}

        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                website_id = next((key for key, value in data.items() if value == url), None)

                print(f"Scraping data from {url} with User-Agent: {user_agent}")

                divs = soup.find_all('div')
                p_tags = soup.find_all('p')

                paragraphs = []
                for div in divs:
                    paragraphs.append(div.get_text(strip=True))

                for p in p_tags:
                    paragraphs.append(p.get_text(strip=True))

                return paragraphs, website_id

            elif response.status_code == 403:
                print(f"403 Forbidden error encountered. Retrying with a different User-Agent...")
            else:
                print(f"Failed to retrieve {url}: {response.status_code}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return []

    print(f"Failed to scrape {url} after 3 attempts.")
    return []


# Function to extract future predictions from the scraped text
def extract_future_predictions(paragraphs, keywords_data):
    result = []

    matcher = Matcher(nlp.vocab)

    future_pattern = [
        {"tag": "MD", "lemma": {"in": keywords_data["modal_verbs"]}},
        {"pos": "VERB", "tag": {"in": ["VB", "VBP", "VBZ"]}},
        {"dep": "prep", "text": {"in": keywords_data["time_expressions"]}},
    ]

    matcher.add("FUTURE_PREDICTION", [future_pattern])

    for para in paragraphs:
        doc = nlp(para)

        matches = matcher(doc)
        future_detected = False

        if matches:
            future_detected = True

        keyword_detected = any(word in para.lower() for word in keywords_data["keywords"])

        if future_detected or keyword_detected:
            result.append(para)

    return result


# Function to save predictions to the SQLite database
def save_predictions_to_db(predictions, website_id):
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
                f"Deleting irrelevant prediction: {text[:30]}...")  # Print the start of the irrelevant text for debugging
            cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))

    conn.commit()
    conn.close()


def main():
    create_db()

    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        user_agents = load_json(user_agents_file_path)

        for key, link in data.items():
            print(f"Scraping link: {link}")

            paragraphs, website_id = scrape_data(link, user_agents, data)

            if paragraphs:
                keywords_data = load_json('keywords.json')

                predicted_paragraphs = extract_future_predictions(paragraphs, keywords_data)

                valid_predictions = []
                for prediction in predicted_paragraphs:
                    doc = nlp(prediction)

                    future_detected = False
                    for token in doc:
                        if token.tag_ in ["MD", "VB", "VBP", "VBZ"] and token.text.lower() in ["will", "shall",
                                                                                               "expect", "predict"]:
                            future_detected = True
                            break

                    if future_detected:
                        valid_predictions.append(prediction)

                if valid_predictions:
                    save_predictions_to_db(valid_predictions, website_id)




    except json.JSONDecodeError:
        print("Error decoding JSON file")


if __name__ == "__main__":
    main()
