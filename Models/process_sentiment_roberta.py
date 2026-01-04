import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from scipy.special import softmax
from torch.cuda.amp import autocast

# CONFIG
INPUT_FILES = [
    "../DataBases/Bitcoin_tweets.csv",
    "../DataBases/Bitcoin_tweets_dataset_2.csv"
]
OUTPUT_FILE = "../DataBases/ROBERTA_SCORES.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# PERFORMANCE SETTINGS
BATCH_SIZE = 128
MAX_LEN = 64
NUM_WORKERS = 0
CHUNK_SIZE = 50000


class TweetDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return str(self.texts[idx])


def process_chunk(df_chunk, model, tokenizer, device, mode, header):
    dataset = TweetDataset(df_chunk['text'].tolist())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    buffer_dates = df_chunk['date'].tolist()
    batch_buffer = []

    with torch.no_grad():
        for i, batch_texts in enumerate(tqdm(loader, desc="Processing Chunk", leave=False)):
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(
                device)

            with torch.amp.autocast('cuda'):
                outputs = model(**inputs)

            scores = outputs.logits.cpu().numpy()
            scores = softmax(scores, axis=1)

            for j, score in enumerate(scores):
                compound = score[2] - score[0]  # Pos - Neg

                global_idx = (i * BATCH_SIZE) + j
                if global_idx < len(buffer_dates):
                    batch_buffer.append({
                        'Date': buffer_dates[global_idx],
                        'Roberta_Score': compound
                    })

    if batch_buffer:
        pd.DataFrame(batch_buffer).to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)
        return True
    return False


def main():

    # 1. SETUP GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. LOAD AI MODEL
    print(f"Loading Model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 3. DETERMINE START POINT
    total_processed = 0
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                total_processed = sum(1 for _ in f) - 1
            if total_processed < 0: total_processed = 0
            print(f"Found existing progress: {total_processed} tweets done.")
        except:
            total_processed = 0

    # 4. STREAMING LOOP
    current_offset = 0
    header_needed = (total_processed == 0)
    mode = 'a'

    for file_path in INPUT_FILES:
        print(f"Processing File: {file_path}")

        try:
            chunk_iterator = pd.read_csv(
                file_path,
                chunksize=CHUNK_SIZE,
                usecols=lambda c: c.lower() in ['date', 'text', 'cleaned_text', 'timestamp', 'user_verified'],
                on_bad_lines='skip',
                encoding='utf-8',
                engine='python'
            )
        except Exception as e:
            print(f"CRITICAL ERROR reading {file_path}: {e}")
            continue

        for chunk_idx, df_chunk in enumerate(chunk_iterator):

            df_chunk.columns = map(str.lower, df_chunk.columns)
            if 'timestamp' in df_chunk.columns: df_chunk.rename(columns={'timestamp': 'date'}, inplace=True)
            if 'cleaned_text' in df_chunk.columns: df_chunk.rename(columns={'cleaned_text': 'text'}, inplace=True)

            df_chunk = df_chunk.dropna(subset=['text'])
            rows_in_chunk = len(df_chunk)

            if current_offset + rows_in_chunk <= total_processed:
                current_offset += rows_in_chunk
                print(f"  Skipping chunk {chunk_idx}...", end='\r')
                continue

            if current_offset < total_processed:
                start_slice = total_processed - current_offset
                df_chunk = df_chunk.iloc[start_slice:]

            wrote_data = process_chunk(df_chunk, model, tokenizer, device, mode, header_needed)

            if wrote_data:
                header_needed = False

            current_offset += rows_in_chunk


            del df_chunk
            import gc
            gc.collect()

    print("\nAll files processed.")


if __name__ == '__main__':
    main()