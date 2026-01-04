import pandas as pd
import sys

# CONFIG
BASELINE_FILE = "../Databases/DIPLOMA_BASELINE_DATA.csv"
SENTIMENT_FILE = "../Databases/HOURLY_SENTIMENT_SCORES_VADER.csv"
OUTPUT_FILE = "../Databases/DIPLOMA_FUSION_DATA_VADER.csv"

print("--- MERGING DATASETS ---")

# 1. Load Price Data (Baseline)
try:
    df_price = pd.read_csv(BASELINE_FILE)
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_price = df_price.set_index('Date')
    print(f"Price Data Loaded: {len(df_price)} rows")
except FileNotFoundError:
    print("ERROR: Could not find DIPLOMA_BASELINE_DATA.csv")
    sys.exit()

# 2. Load Sentiment Data
try:
    df_sent = pd.read_csv(SENTIMENT_FILE)
    df_sent['Date'] = pd.to_datetime(df_sent['Date'])
    df_sent = df_sent.set_index('Date')

    # === FIX: REMOVE FUTURE DATA ===
    df_sent = df_sent[df_sent.index.year <= 2024]

    print(f"Sentiment Data Loaded: {len(df_sent)} rows (2025 outlier removed)")
except FileNotFoundError:
    print("ERROR: Could not find HOURLY_SENTIMENT_SCORES_VADER.csv")
    sys.exit()

# 3. Merge (Left Join)
print("Merging tables...")
df_merged = df_price.join(df_sent, how='left')

# 4. INTELLIGENT FILLING (Forward Fill)
# We fill gaps up to 24 hours. If Twitter is silent, we assume sentiment hasn't changed.
df_merged['Sentiment'] = df_merged['Sentiment'].ffill(limit=24)

# Fill any remaining gaps (longer than 24h) with 0.0 (Neutral)
missing_before = df_merged['Sentiment'].isna().sum()
df_merged['Sentiment'] = df_merged['Sentiment'].fillna(0.0)
df_merged['Tweet_Volume'] = df_merged['Tweet_Volume'].fillna(0)

print(f"   -> Filled {missing_before} empty hours using Forward Fill/Zero.")

# 5. Save
df_merged.to_csv(OUTPUT_FILE)
print("=" * 50)
print(f"Final Fusion Data saved to: {OUTPUT_FILE}")
print("=" * 50)