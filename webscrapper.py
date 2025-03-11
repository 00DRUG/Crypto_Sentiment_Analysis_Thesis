import sqlite3
import requests
import spacy
import json
import random
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# SQLite database name
DB_NAME = "crypto_predictions.db"

# Replace with the path to your local JSON files
json_file_path = 'links.json'  # Links file
user_agents_file_path = 'user_agents.json'  # User Agents file


# Function to load user agents from a JSON file
def load_user_agents(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Function to create the SQLite database and table if they don't exist
def create_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


# Function to scrape data from a given URL with user-agent rotation
def scrape_data(url, user_agents):
    for attempt in range(3):
        browser_category = random.choice(list(user_agents.keys()))
        user_agent = random.choice(user_agents[browser_category])
        headers = {'User-Agent': user_agent}

        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                # Parse the page using BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                print(f"Scraping data from {url} with User-Agent: {user_agent}")

                # Example filters: Get all divs with a specific class or p tags
                divs = soup.find_all('div', class_='your-div-class')  # Modify with your class
                p_tags = soup.find_all('p')  # All <p> tags

                # Extract text from divs and p tags
                paragraphs = []
                for div in divs:
                    paragraphs.append(div.get_text(strip=True))

                for p in p_tags:
                    paragraphs.append(p.get_text(strip=True))

                return paragraphs  # Return all extracted paragraphs

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
def extract_future_predictions(paragraphs):
    result = []
    keywords = ["prediction", "forecast", "expected", "will reach", "by 2025", "price target"]

    for para in paragraphs:
        doc = nlp(para)
        future_detected = any(
            token.tag_ in ["MD", "VB", "VBP", "VBZ"] and token.text.lower() in ["will", "shall", "expect", "predict"]
            for token in doc)
        keyword_detected = any(word in para.lower() for word in keywords)

        if future_detected or keyword_detected:
            result.append(para)

    return result


# Function to save predictions to the SQLite database
def save_predictions_to_db(predictions):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for pred in predictions:
        cursor.execute("INSERT INTO predictions (text) VALUES (?)", (pred,))

    conn.commit()
    conn.close()


# Function to delete old predictions (older than 30 days)
def delete_old_predictions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Use timezone-aware datetime to avoid deprecation warning
    threshold_date = datetime.now(timezone.utc) - timedelta(days=30)
    cursor.execute("DELETE FROM predictions WHERE timestamp < ?", (threshold_date,))

    conn.commit()
    conn.close()


# Function to delete irrelevant predictions (not in future tense)
def delete_irrelevant_predictions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT id, text FROM predictions")
    predictions = cursor.fetchall()

    for pred_id, text in predictions:
        doc = nlp(text)
        future_detected = any(
            token.tag_ in ["MD", "VB", "VBP", "VBZ"] and token.text.lower() in ["will", "shall", "expect", "predict"]
            for token in doc)

        if not future_detected:
            cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))

    conn.commit()
    conn.close()


# Main function to run the scraper, extract predictions, and store them in the database
def main():
    create_db()  # Ensure the database is created

    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)  # Load links from the JSON file

        user_agents = load_user_agents(user_agents_file_path)  # Load user agents

        # Iterate through the links with numeric keys (1, 2, 3, 4, etc.)
        for key, link in data.items():
            print(f"Scraping link: {link}")

            paragraphs = scrape_data(link, user_agents)  # Scrape the page

            if paragraphs:
                # Filter relevant predictions
                predicted_paragraphs = extract_future_predictions(paragraphs)

                if predicted_paragraphs:
                    save_predictions_to_db(predicted_paragraphs)  # Save to database

            # Optionally, clean the database
            delete_old_predictions()
            delete_irrelevant_predictions()

    except json.JSONDecodeError:
        print("Error decoding JSON file")


if __name__ == "__main__":
    main()
