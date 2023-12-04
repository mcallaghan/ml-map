#!/usr/bin/env python

import argparse
from mlmap import load_data, hf_tokenize_data
from mlmap.transformer_utils import CustomTrainer, compute_metrics, CustomTrainingArguments, optuna_hp_space
from transformers import AutoModelForSequenceClassification
from optuna.storages import RDBStorage
from optuna.study import delete_study
from sklearn.model_selection import KFold
import os
import pandas as pd
from pathlib import Path
import numpy as np
import json

def pipeline_predict():
    parser = argparse.ArgumentParser(description='Make predictions with a pretrained model')
    parser.add_argument('-m', type=str, dest='model_name',
                        required=True)
    parser.add_argument('-y', type=str, dest='y_prefix',
                       default='INCLUDE')
    args = parser.parse_args()    


    # Load our unlabelled data, and our labels
    df = pd.read_feather('data/documents.feather')
    labels = pd.read_feather('data/labels.feather')

    # We only need predictions where we have no labels
    df = df[~df["id"].isin(labels["id"])]
    # We will not make predictions when we have missing data
    df = df.dropna(subset=["title","abstract"]).reset_index(drop=True)
    df["text"] = df["title"] + " " + df["abstract"]
    # We just want the id and text columns
    df = df[["id","text"]]

    # What did we call this study when we ran it
    study_name = f"{args.model_name.replace('/','__')}__{args.y_prefix}"
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        f'results/{study_name}_final_model'
    )
    trainer = CustomTrainer(
        model=model
    )

    # Load the shape of what the model predicts, so we can initialise an empty array with a similar shape
    outer_preds = np.load(f'results/predictions/{study_name}__0__outer_predictions.npy')
    if outer_preds.ndim==1:
        y_pred = np.zeros(df.shape[0])
        binary=True
    else:
        y_pred = np.zeros((df.shape[0],outer_preds.shape[1]))
        binary=False
        
    # To make the process more memory efficient, we will make predictions one batch at a time
    batch_size = 10000 # this number can probably increase depending on how much memory you have
    for i, chunk in df.groupby(np.arange(len(df))//batch_size):
        print(f"predicting batch {i}")
        chunk_ds = hf_tokenize_data(chunk, args.model_name)
        y_pred[chunk.index] = trainer.predict_proba(chunk_ds, binary=binary)
    # Save the predictions
    np.save(f"results/predictions/{study_name}__predictions", y_pred)
    np.save(f"results/predictions/{study_name}__ids", df["id"].values)

if __name__ == '__main__':
    pipeline_predict()