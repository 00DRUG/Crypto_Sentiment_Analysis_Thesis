import csv
import sys
import datetime
from dateutil import parser

csv.field_size_limit(sys.maxsize)
files = [
    "../DataBases/Bitcoin_tweets_1.csv",
    "../DataBases/Bitcoin_tweets_dataset_2.csv"
]

global_min = None
global_max = None

print("--- ROBUST DATE SCANNER ---")

for filepath in files:
    print(f"\nScanning {filepath}...")

    current_file_min = None
    current_file_max = None
    row_count = 0

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)

            header = next(reader, None)
            if not header:
                print("   -> Empty file.")
                continue

            header = [h.strip().lower() for h in header]

            try:
                # Try to find 'date' or 'timestamp' column
                if 'date' in header:
                    date_idx = header.index('date')
                elif 'timestamp' in header:
                    date_idx = header.index('timestamp')
                else:
                    print(f"   -> Error: Could not find 'date' column in header: {header}")
                    continue
            except ValueError:
                continue

            # 2. Scan Rows
            for row in reader:
                row_count += 1

                if row_count % 500000 == 0:
                    print(f"   ... processed {row_count} rows ...")

                try:
                    if len(row) > date_idx:
                        date_str = row[date_idx]

                        if date_str and len(date_str) > 5:
                            dt = parser.parse(date_str)

                            if current_file_min is None or dt < current_file_min:
                                current_file_min = dt
                            if current_file_max is None or dt > current_file_max:
                                current_file_max = dt

                except Exception:
                    continue

        print(f"   -> File Result: {current_file_min} to {current_file_max}")

        if global_min is None or (current_file_min and current_file_min < global_min):
            global_min = current_file_min
        if global_max is None or (current_file_max and current_file_max > global_max):
            global_max = current_file_max

    except Exception as e:
        print(f"   -> CRITICAL ERROR reading file: {e}")

print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
if global_min and global_max:
    print(f"Earliest Tweet: {global_min}")
    print(f"Latest Tweet:   {global_max}")
    print("-" * 50)
else:
    print("Could not find any valid dates.")
