import os
import json
import requests
import tweepy
import sqlite3
from datetime import datetime

# Register datetime adapters
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())  # Convert datetime to ISO 8601 string
sqlite3.register_converter("DATETIME", lambda s: datetime.fromisoformat(s.decode("utf-8")))

# Database setup
db_path = "tweets.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS tweets (
    tweet_id TEXT PRIMARY KEY,
    author_id TEXT,
    text TEXT,
    created_at TEXT,
    likes INTEGER,
    retweets INTEGER,
    hashtags TEXT,
    lang TEXT,
    media_url TEXT,
    media_local_path TEXT
);
""")
conn.commit()


def download_image(media_url, save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(media_url)
    local_path = os.path.join(save_dir, filename)

    response = requests.get(media_url)
    if response.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(response.content)
        return local_path
    return None


def insert_tweet(tweet):
    media_local_path = None

    if tweet.get("media_url"):
        media_local_path = download_image(tweet["media_url"])

    cursor.execute("""
    INSERT OR REPLACE INTO tweets (tweet_id, author_id, text, created_at, likes, retweets, hashtags, lang, media_url, media_local_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        tweet["tweet_id"],
        tweet["author_id"],
        tweet["text"],
        tweet["created_at"],
        tweet["likes"],
        tweet["retweets"],
        ",".join(tweet["hashtags"]) if tweet.get("hashtags") else None,
        tweet["lang"],
        tweet.get("media_url"),
        media_local_path
    ))
    conn.commit()


with open("twitter_keys.json") as infile:
    json_obj = json.load(infile)
    token = json_obj["bearer_token"]
    client = tweepy.Client(bearer_token=token)

username = "Crypto_Twittier"
keyword = "🚨Prediction🚨"


def fetch_tweets(username, keyword, max_results=10):
    try:
        query = f"from:{username} {keyword}"
        response = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["created_at", "public_metrics", "entities", "lang"]
        )
        for tweet_data in response.data:
            tweet = {
                "tweet_id": tweet_data.id,
                "author_id": tweet_data.author_id,
                "text": tweet_data.text,
                "created_at": tweet_data.created_at,
                "likes": tweet_data.public_metrics.get('like_count', 0),
                "retweets": tweet_data.public_metrics.get('retweet_count', 0),
                "hashtags": [hashtag['tag'] for hashtag in tweet_data.entities.get('hashtags', [])],
                "lang": tweet_data.lang,
                "media_url": tweet_data.entities.get('media', [{}])[0].get('media_url')
            }
            insert_tweet(tweet)
        print(f"Successfully fetched and stored {len(response.data)} tweets.")
    except tweepy.errors.TooManyRequests:
        print("Rate limit hit. Please wait before retrying.")
    except Exception as e:
        print(f"An error occurred: {e}")


fetch_tweets(username, keyword)


def verify_data():
    cursor.execute("SELECT COUNT(*) FROM tweets")
    count = cursor.fetchone()[0]
    print(f"{count} tweets are stored in the database.")


verify_data()

conn.close()
print("Tweets stored successfully in the database.")
