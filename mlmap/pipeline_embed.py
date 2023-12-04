#!/usr/bin/env python

import argparse
from mlmap import load_data, hf_tokenize_data
from mlmap.transformer_utils import CustomTrainer, compute_metrics, CustomTrainingArguments, optuna_hp_space
from transformers import AutoModel, AutoTokenizer
from optuna.storages import RDBStorage
from optuna.study import delete_study
from sklearn.model_selection import KFold
from torch import cuda
import os
import pandas as pd
from pathlib import Path
import numpy as np
import json

def pipeline_embed():
    parser = argparse.ArgumentParser(description='Make predictions with a pretrained model')
    parser.add_argument('-m', type=str, dest='model_name',
                        required=True)
    args = parser.parse_args()    

    device = "cuda:0" if cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="transformers")
    
    model_str = args.model_name.replace('/','__')
    # Load our final dataset
    df = pd.read_feather('data/final_dataset.feather')

    embeddings = np.zeros((df.shape[0], 768))
    chunk_size = 8
    for i, group in df.groupby(np.arange(len(df))//chunk_size):
        print(i)
        
        title_abs = group.apply(lambda x: x["title"] + tokenizer.sep_token + x["abstract"], axis=1)
        # preprocess the input
        inputs = tokenizer(list(title_abs.values), padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        
        # inference
        model = AutoModel.from_pretrained(args.model_name, cache_dir="transformers").to(device)
        result = model(**inputs)
    
        # take the first token ([CLS] token) in the batch as the embedding
        batch_embeddings = result.last_hidden_state[:, 0, :]
        embeddings[group.index] = batch_embeddings.detach().cpu().numpy()
        del batch_embeddings
        del model
        del result
        cuda.empty_cache()

    np.save(f"results/{model_str}__embeddings", embeddings)
    np.save(f"results/{model_str}__embeddings_ids", df["id"].values)

if __name__ == '__main__':
    pipeline_embed()