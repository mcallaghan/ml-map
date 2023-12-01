from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

def hf_tokenize_data(df, model_name):
    # Turn the data into a Huggingface dataset
    d = {
        'text': df['text']
    }
    if 'labels' in df.columns:
        d['labels'] = df['labels']
    dataset = Dataset.from_dict(d)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding='max_length'
        ), batched=True
    )
    dataset.set_format("torch")
    # Remove the now redundant text column
    return dataset.remove_columns("text")
