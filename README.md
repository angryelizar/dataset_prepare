# Dataset Prepare

A tool for preparing a sentiment analysis dataset.

## What it does

1. Downloads [k1tub/sentiment_dataset](https://huggingface.co/datasets/k1tub/sentiment_dataset)
2. Analyzes token distribution (text lengths)
3. Splits into train/test (80/20) with stratification by class
4. Saves the resulting dataset

## Resulting dataset

[angryelizar/sentiment_dataset_splitted](https://huggingface.co/datasets/angryelizar/sentiment_dataset_splitted)

## Usage

```bash
uv run dataset-prepare
```