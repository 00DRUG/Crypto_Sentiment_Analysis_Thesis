import pandas as pd
import os

# CONFIG
FILE_PATH = "./DataBases/ROBERTA_SCORES.csv"


if not os.path.exists(FILE_PATH):
    print("ERROR: File not found!")
else:
    # 1. Check File Size
    file_size = os.path.getsize(FILE_PATH) / (1024 * 1024)
    print(f"File Size:      {file_size:.2f} MB")

    # 2. Load Data
    print("Loading data...")
    df = pd.read_csv(FILE_PATH)

    # 3. Basic Stats
    total_rows = len(df)
    print(f"Total Tweets:   {total_rows:,}")

    if total_rows > 0:
        # Convert date to check range
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        min_date = df['Date'].min()
        max_date = df['Date'].max()

        print(f"Date Range:     {min_date}  <--->  {max_date}")

        # 4. Check Sentiment Scores
        avg_score = df['Roberta_Score'].mean()
        print(f"saved_score (Avg): {avg_score:.4f} (Should be close to 0, e.g. -0.1 to 0.1)")

        print("\n--- FIRST 5 ROWS ---")
        print(df.head())

        print("\n--- LAST 5 ROWS ---")
        print(df.tail())

        # 5. Validity Check
        zeros = (df['Roberta_Score'] == 0.0).sum()
        print(f"\nNOTE: {zeros} rows have exactly 0.0 score (Neutral).")
        if total_rows > 100000:
            print("STATUS: You have a big dataset.")
    else:
        print("STATUS: The file is empty.")