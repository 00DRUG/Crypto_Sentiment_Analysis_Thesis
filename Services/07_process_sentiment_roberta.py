import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.special import softmax

INPUT_FILES = ["../DataBases/Bitcoin_tweets.csv", "../DataBases/Bitcoin_tweets_dataset_2.csv"]
OUTPUT_FILE = "../DataBases/ROBERTA_SCORES.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
BATCH_SIZE = 64


class TweetDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return str(self.texts[idx])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    all_results = []

    for fpath in INPUT_FILES:
        df = pd.read_csv(fpath, on_bad_lines='skip')
        df = df.dropna(subset=['text'])

        dataset = TweetDataset(df['text'].tolist())
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        dates = df['date'].tolist()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
                outputs = model(**inputs)
                scores = outputs.logits.cpu().numpy()
                scores = softmax(scores, axis=1)

                for j, s in enumerate(scores):
                    idx = i * BATCH_SIZE + j
                    if idx < len(dates):
                        all_results.append({
                            'Date': dates[idx],
                            'Score': s[2] - s[0]
                        })

    pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False)


if __name__ == '__main__':
    main()