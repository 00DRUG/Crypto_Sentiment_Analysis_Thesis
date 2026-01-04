import pandas as pd
import os

# --- CONFIGURATION ---
file_name = "../Databases/Bitcoin_tweets.csv"
date_column = 'date'

try:
    print(f"Reading {file_name}...")

    df = pd.read_csv(file_name, lineterminator='\n', low_memory=False)

    df.columns = df.columns.str.strip()

    if date_column not in df.columns:
        print(f" Error: Column '{date_column}' not found.")
        print("Available columns:", df.columns.tolist())
    else:
        print("Processing dates (this may take a moment)...")

        # Convert to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Drop invalid dates
        df = df.dropna(subset=[date_column])

        # Group by Year-Month
        monthly_counts = df.groupby(df[date_column].dt.to_period('M')).size()

        # Sort chronologically
        monthly_counts = monthly_counts.sort_index()

        print("\n--- Tweet Counts by Year-Month ---")
        print(monthly_counts)
        print(f"\nTotal tweets counted: {monthly_counts.sum()}")

except FileNotFoundError:
    print(f" Error: Could not find '{file_name}'")
    print(f"Your current working folder is: {os.getcwd()}")

#Bitcoin_tweets_1.csv
#--- Tweet Counts by Year-Month ---
#date
#2021-02     44443
#2021-03      4140
#2021-04     58060
#2021-05     21782
#2021-06    125795
#2021-07    466079
#2021-08    488987
#2021-09     23510
#2021-10    351796
#2021-11    359630
#2021-12     55301
#2022-01    260087
#2022-02     79475
#2022-03    360058
#2022-04    417155
#2022-05    356018
#2022-06    346212
#2022-07    193654
#2022-08     41925
#2022-09    182046
#2022-10    146753
#2022-11    202262
#2022-12     43233
#2023-01     60887
#Freq: M, dtype: int64

#Total tweets counted: 4689288

#Bitcoin_tweets_dataset_2.csv
#--- Tweet Counts by Year-Month ---
#date
#2023-02     63645
#2023-03    106175
#Freq: M, dtype: int64

#Total tweets counted: 169820