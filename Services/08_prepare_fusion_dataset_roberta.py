import pandas as pd
import sys
import os

# CONFIG
ROBERTA_FILE = "../DataBases/ROBERTA_SCORES.csv"

PRICE_FILE = "../DataBases/DIPLOMA_BASELINE_DATA.csv"

# 3. The Output
OUTPUT_FILE = "../Databases/DIPLOMA_FUSION_DATA_ROBERTA.csv"

print("--- MERGING DATASETS ---")

# 1. Load the Scores
print(f"Loading {ROBERTA_FILE}...")
if not os.path.exists(ROBERTA_FILE):
    print(f" Error: {ROBERTA_FILE} not found!")
    sys.exit()

# Load raw scores
df_rob = pd.read_csv(ROBERTA_FILE, low_memory=False)

# For the corrupted dates - data
print("Cleaning bad dates...")
df_rob['Date'] = pd.to_datetime(df_rob['Date'], errors='coerce')

# Delete the rows where the Date is broken
initial_len = len(df_rob)
df_rob = df_rob.dropna(subset=['Date'])
print(f"Removed {initial_len - len(df_rob)} bad rows. Valid rows: {len(df_rob)}")

# 2. Aggregate by Hour
print("Aggregating to Hourly Averages...")
df_hourly = df_rob.set_index('Date').resample('h').agg({
    'Roberta_Score': 'mean'
})

# Calculate Tweet Volume
df_volume = df_rob.set_index('Date').resample('h').size().rename("Tweet_Volume")

# Join them
df_final_sentiment = df_hourly.join(df_volume)

# 3. Merge with Price Data
print(f"Merging with Price Data ({PRICE_FILE})...")

if not os.path.exists(PRICE_FILE):
    print(f" Warning: Price file not found!")
    df_merged = df_final_sentiment
else:
    df_price = pd.read_csv(PRICE_FILE)
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_price = df_price.set_index('Date')

    # Merge
    df_merged = df_price.join(df_final_sentiment, how='left')

# 4. Cleanup
df_merged['Roberta_Score'] = df_merged['Roberta_Score'].fillna(0.0)
df_merged['Tweet_Volume'] = df_merged['Tweet_Volume'].fillna(0)
df_merged = df_merged.rename(columns={'Roberta_Score': 'Sentiment'})

# 5. Save
df_merged.to_csv(OUTPUT_FILE)
print("=" * 50)
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 50)