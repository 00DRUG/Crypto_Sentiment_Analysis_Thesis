import csv
import sys
import pandas as pd
from dateutil import parser
from collections import defaultdict

# CONFIG
INPUT_FILES = [
    "../DataBases/Bitcoin_tweets.csv",
    "../DataBases/Bitcoin_tweets_dataset_2.csv"
]
# Windows CSV LIMIT
csv.field_size_limit(2147483647)

print("--- TWEET DISTRIBUTION DIAGNOSTIC ---")

monthly_counts = defaultdict(int)
total_read = 0
total_parsed = 0
total_errors = 0

for filepath in INPUT_FILES:
    print(f"\nAnalyzing file: {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader, None)

            if not header: continue
            header = [h.strip().lower() for h in header]

            # Find date column
            try:
                date_idx = header.index('date') if 'date' in header else header.index('timestamp')
            except ValueError:
                print("   -> ERROR: Could not find 'date' or 'timestamp' column.")
                continue

            for row in reader:
                total_read += 1
                if total_read % 1000000 == 0:
                    print(f"   ... scanned {total_read} rows ...")

                if len(row) <= date_idx: continue

                date_str = row[date_idx]

                try:
                    # Parse Date
                    dt = parser.parse(date_str)

                    # Create Key (Year-Month)
                    month_key = dt.strftime("%Y-%m")
                    monthly_counts[month_key] += 1
                    total_parsed += 1

                except Exception:
                    total_errors += 1

    except Exception as e:
        print(f"   -> CRITICAL ERROR: {e}")

print("\n" + "=" * 50)
print("FINAL DISTRIBUTION (Tweets per Month)")
print("=" * 50)

# Sort by date
sorted_keys = sorted(monthly_counts.keys())

for key in sorted_keys:
    count = monthly_counts[key]
    print(f"{key}: {count:,} tweets")

print("-" * 50)
print(f"Total Parsed: {total_parsed}")
print(f"Total Date Errors: {total_errors}")
print("=" * 50)