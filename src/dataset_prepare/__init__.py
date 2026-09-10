from transformers import AutoTokenizer
from datasets import load_dataset, ClassLabel
import numpy as np
import pandas as pd

def main():
    print("App is running...\n")
    dataset = download_and_check_dataset()
    # tokenized_dataset = calculate_tokens_per_row(dataset)
    # analyze_token_distribution(tokenized_dataset)
    splitted_dataset = split_dataset(dataset)





def download_and_check_dataset():
    print("Downloading 'Russian Sentiment Analysis Dataset' dataset...")
    dataset = load_dataset("k1tub/sentiment_dataset")
    train = dataset["train"]
    print("Dataset downloaded successfully!\n")
    print(f"Number of rows: {len(train)}")
    print("Printing first 3 rows of dataset...")
    for text in train[:3]["text"]:
        print(f"{text}\n")
    return dataset

def calculate_tokens_per_row(dataset):
    # Dataset will be also used with ai-forever/ruRoberta-large, but the bias in tokenization is not significant
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base")
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], add_special_tokens=False, truncation=False, padding=False,), batched=True)
    # Calculate the number of tokens per row
    tokenized_dataset = tokenized_dataset.map(lambda x: {"num_tokens": [len(tokens) for tokens in x["input_ids"]]}, batched=True)
    part = tokenized_dataset["train"][:3]
    print("Printing first 3 text of tokenized dataset... + number of tokens per row")

    for text, num_tokens in zip(part["text"], part["num_tokens"]):
        print(f"Text: {text}\nNumber of tokens: {num_tokens}\n")
    return tokenized_dataset

def analyze_token_distribution(tokenized_dataset):
    train = tokenized_dataset["train"]
    num_tokens = train["num_tokens"]
    labels = train["label"]

    print("General statistics of the number of tokens per row:")
    print(f"  count: {len(num_tokens)}")
    print(f"  mean:  {np.mean(num_tokens):.1f}")
    print(f"  std:   {np.std(num_tokens):.1f}")
    print(f"  min:   {np.min(num_tokens)}")
    print(f"  max:   {np.max(num_tokens)}")

    print("\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        value = np.percentile(num_tokens, p)
        print(f"  p{p}: {value:.0f} tokens")

    # 99p = 323 tokens, so we can set up the max length to 350 tokens, which is a good compromise between keeping most of the data and not having too long sequences for training

    print("\nPercentiles by class (label):")
    df = pd.DataFrame({"num_tokens": num_tokens, "label": labels})
    print(df.groupby("label")["num_tokens"].describe())

    # Look at the shortest texts (<= 6 tokens)
    # TLDR: Still a good texts, we can keep them in the dataset
    # shortest_texts = train.filter(lambda x: x["num_tokens"] <= 6);
    # for text in shortest_texts["text"]:
    #     print(f"Shortest text: {text}\n")

    # Look at the shortest texts (<= 3 tokens)
    # TLDR: Still a good texts, we can keep them in the dataset
    # shortest_texts = train.filter(lambda x: x["num_tokens"] <= 2);
    # for text in shortest_texts["text"]:
    #     print(f"Shortest text: {text}\n")

    return df

def split_dataset(dataset):
    dataset = dataset["train"]
    class_labels = ClassLabel(names=["neutral", "positive", "negative"])
    dataset = dataset.cast_column("label", class_labels)
    splitted_dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)

    print(f"\nDataset splitted into train and test sets with 80% and 20% of the data respectively.")
    print(f"Number of rows in train set: {len(splitted_dataset['train'])}")
    print(f"Number of rows in test set: {len(splitted_dataset['test'])}")
    splitted_dataset.save_to_disk("../data/sentiment_dataset")