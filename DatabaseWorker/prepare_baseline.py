import pandas as pd
import sys
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_PRICE_FILE = "../DataBases/btc_1h_data_2018_to_2025.csv"
OUTPUT_FILE = "../DataBases/DIPLOMA_BASELINE_DATA.csv"

# Dates for identical DB
START_DATE = "2021-02-05 10:00:00"
END_DATE = "2023-03-05 23:59:59"

print("--- STEP 1: PREPARING BASELINE DATA ---")

print(f"Loading {INPUT_PRICE_FILE}...")
try:
    # Use needed columns
    cols_to_use = ['Open time', 'Close', 'Volume']

    df = pd.read_csv(INPUT_PRICE_FILE, usecols=cols_to_use)

    # Trim whitespace
    df.columns = df.columns.str.strip()

except ValueError as e:
    print(f"ERROR: Column mismatch. Check your CSV header. Details: {e}")
    sys.exit()
except FileNotFoundError:
    print(f"ERROR: Could not find '{INPUT_PRICE_FILE}'.")
    sys.exit()

# 2. Rename & Parse Dates
print("Parsing dates...")
df.rename(columns={'Open time': 'Date'}, inplace=True)

# Parse the date format: "2018-01-01 00:00:00.000000"
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows where date failed to parse
df = df.dropna(subset=['Date'])

# Set Index
df = df.set_index('Date')
df = df.sort_index()

# 3. CUT THE DATA
print(f"Cutting data to range: {START_DATE} to {END_DATE}...")
original_len = len(df)
df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
new_len = len(df)

print(f"   -> Reduced from {original_len} to {new_len} rows.")

if df.empty:
    print("CRITICAL ERROR: The date filter removed all rows! Check if your CSV dates match the filter year.")
    sys.exit()

# 4. Feature Engineering
print("Calculating Indicators...")

# Ensure columns are numeric
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

# Indicators
df['Returns'] = df['Close'].pct_change()
df['Vol_Change'] = df['Volume'].pct_change()
df['SMA_24'] = df['Close'].rolling(window=24).mean()
df['Dist_SMA'] = df['Close'] - df['SMA_24']

# Volatility (Standard Deviation of last 5 hours)
df['Volatility'] = df['Close'].rolling(window=5).std()

# 5. Create TARGET Variable
# Target = 1 if Next Hour Close > Current Hour Close
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# 6. Clean Up
df = df.dropna()

# 7. Save
print(f"Saving to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE)

print("=" * 60)
print("SUCCESS! Baseline data prepared.")
print(f"Rows ready for training: {len(df)}")
print("=" * 60)