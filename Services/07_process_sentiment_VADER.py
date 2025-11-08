import csv
import sys
import pandas as pd
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dateutil import parser

# CONFIG
INPUT_FILES = [
    "../DataBases/Bitcoin_tweets.csv",
    "../DataBases/Bitcoin_tweets_dataset_2.csv"
]
OUTPUT_FILE = "../Databases/HOURLY_SENTIMENT_SCORES_VADER.csv"

# 1. Setup VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
csv.field_size_limit(2147483647)

# 2. Compile Regex Patterns
# Removes URLs (http...)
pat_url = re.compile(r'http\S+|www\.\S+')
# Removes @mentions
pat_mention = re.compile(r'@\w+')
# Removes the '#' symbol but keeps the word
pat_hash = re.compile(r'#')

# 3. Spam Filter  - store hash of tweets to avoid duplicates
seen_tweet_hashes = set()


def clean_tweet(text):
    # 1. Remove URLs
    text = pat_url.sub('', text)
    # 2. Remove @mentions
    text = pat_mention.sub('', text)
    # 3. Remove '#' symbol
    text = pat_hash.sub('', text)
    # 4. Remove extra whitespace
    return " ".join(text.split())


print("--- STARTING CLEANED SENTIMENT ANALYSIS ---")

hourly_data = {}  # { "2021-02-05 10:00:00": [total_score, tweet_count] }
skipped_bots = 0
processed_count = 0

for filepath in INPUT_FILES:
    print(f"\nProcessing {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader, None)

            if not header: continue
            header = [h.strip().lower() for h in header]

            try:
                date_idx = header.index('date') if 'date' in header else header.index('timestamp')
                text_idx = header.index('text')
            except ValueError:
                print(f"   -> Skipping file, missing columns: {header}")
                continue

            for row in reader:
                try:
                    if len(row) <= text_idx: continue

                    raw_text = row[text_idx]

                    # === STEP 1: SPAM FILTER ===
                    # Simple check: If we saw this exact text before, skip it.
                    text_hash = hash(raw_text)
                    if text_hash in seen_tweet_hashes:
                        skipped_bots += 1
                        continue
                    seen_tweet_hashes.add(text_hash)

                    # === STEP 2: CLEANING ===
                    clean_text = clean_tweet(raw_text)

                    # Skip empty tweets
                    if len(clean_text) < 3:
                        continue

                    # === STEP 3: DATE PARSING ===
                    date_str = row[date_idx]
                    dt = parser.parse(date_str)
                    # Round down to hour
                    dt_hour = dt.replace(minute=0, second=0, microsecond=0)
                    key = str(dt_hour)

                    # === STEP 4: VADER SCORING ===
                    score = sia.polarity_scores(clean_text)['compound']

                    # === STEP 5: AGGREGATE ===
                    if key not in hourly_data:
                        hourly_data[key] = [0.0, 0]  # [sum, count]

                    hourly_data[key][0] += score
                    hourly_data[key][1] += 1

                    processed_count += 1
                    if processed_count % 100000 == 0:
                        print(f"   ... {processed_count} valid tweets (Skipped {skipped_bots} bots) ...")

                except Exception:
                    continue

    except Exception as e:
        print(f"   -> Error reading file: {e}")

print(f"\nTotal Tweets Analyzed: {processed_count}")
print(f"Total Bots/Duplicates Removed: {skipped_bots}")

# --- SAVE RESULTS ---
print("Saving results...")
results = []
for date_key, values in hourly_data.items():
    total_score = values[0]
    count = values[1]
    avg_score = total_score / count
    results.append({'Date': date_key, 'Sentiment': avg_score, 'Tweet_Volume': count})

df_final = pd.DataFrame(results)
df_final['Date'] = pd.to_datetime(df_final['Date'])
df_final = df_final.sort_values('Date')
df_final.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print(f"Cleaned sentiment saved to: {OUTPUT_FILE}")
print("=" * 60)