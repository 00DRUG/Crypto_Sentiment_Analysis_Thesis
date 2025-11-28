import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import torch

# CONFIG
INPUT_FILE = "../DataBases/Bitcoin_tweets.csv"
OUTPUT_FILE = "../DataBases/ROBERTA_SCORES.csv"
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

df = pd.read_csv(INPUT_FILE, encoding='utf-8', on_bad_lines='skip')
df = df.dropna(subset=['text'])

results = []

print(f"Processing {len(df)} rows...")
for index, row in df.iterrows():
    text = str(row['text'])
    try:
        encoded = tokenizer(text, return_tensors='pt')
        output = model(**encoded)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # Labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
        compound = scores[2] - scores[0]

        results.append({
            "date": row['date'],
            "roberta_score": compound
        })

        if index % 100 == 0:
            print(index)

    except Exception as e:
        print(f"Error: {e}")

# Save
pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
print("Done")