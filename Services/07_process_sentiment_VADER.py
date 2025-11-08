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

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
csv.field_size_limit(2147483647)

pat_url = re.compile(r'http\S+|www\.\S+')
pat_mention = re.compile(r'@\w+')
pat_hash = re.compile(r'#')


def clean_tweet(text):
    text = pat_url.sub('', text)
    text = pat_mention.sub('', text)
    text = pat_hash.sub('', text)
    return " ".join(text.split())


print("Starting VADER analysis...")

hourly_data = {}

for filepath in INPUT_FILES:
    print(f"Reading {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header: continue

            # Simple header search
            header = [h.strip().lower() for h in header]
            try:
                date_idx = header.index('date') if 'date' in header else header.index('timestamp')
                text_idx = header.index('text')
            except:
                continue

            for row in reader:
                try:
                    if len(row) <= text_idx: continue

                    raw_text = row[text_idx]
                    clean_text = clean_tweet(raw_text)

                    if len(clean_text) < 3: continue

                    date_str = row[date_idx]
                    dt = parser.parse(date_str)
                    dt_hour = dt.replace(minute=0, second=0, microsecond=0)
                    key = str(dt_hour)

                    score = sia.polarity_scores(clean_text)['compound']

                    if key not in hourly_data:
                        hourly_data[key] = [0.0, 0]

                    hourly_data[key][0] += score
                    hourly_data[key][1] += 1

                except:
                    continue
    except Exception as e:
        print(e)

# Save
results = []
for date_key, values in hourly_data.items():
    avg = values[0] / values[1]
    results.append({'Date': date_key, 'Sentiment': avg, 'Tweet_Volume': values[1]})

df = pd.DataFrame(results)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.to_csv(OUTPUT_FILE, index=False)
print("Done.")