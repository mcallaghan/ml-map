---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Putting our pipeline together

We now have all the elements we need to come up with a final model, estimate its performance, and use it to make predictions about the data we have not labelled.

Running this pipeline on the full dataset is computationally intensive, so it is written as set of python scripts.

`run_full_pipeline.sh` will run the whole training, evaluation and prediction pipeline for each target variable(s). You can change the model name variable in the script to run the pipeline with different models

`pipeline_train.py` trains and evaluates a given model on a given target variable. It saves the final model, as well as evaluation scores and predictions made for the outer test sets in the `results` directory.

Run `python mlmap/pipeline_train.py -h` to see the possible arguments

`pipeline_predict.py` takes the saved model, and makes predictions for documents that do not have labels

## Trial data

```{code-cell} ipython3
:tags: [remove-cell, thebe-init]

import os
os.chdir('../../../')
```

The results of our trials are stored in `results/trials.db`. We can inspect these as follows:

```{code-cell} ipython3
:tags: [thebe-init]

from mlmap import return_search
db = "results/trials.db"
model_name = "distilroberta-base"
y_prefix = "INCLUDE"
study_name = f"{model_name}__{y_prefix}"
df = return_search(db, study_name)
# Number of trials completed
print(df.number.unique().shape[0])
df.head()
```

We can see how long it took to get to the highest value in our set of trials in each outer fold


```{code-cell} ipython3
:tags: [thebe-init]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7,4))
for fold, group in df.groupby("study_name"):
    f1_max = group.groupby('number')['value'].max().cummax()
    ax.plot(f1_max)

```

## Final model performance

```{code-cell} ipython3
:tags: [thebe-init]

import json
import pandas as pd

model = 'distilroberta-base'

ys = ["INCLUDE","4 -","8 -"]
scores = []
for y in ys:
    for k in range(3):
        study_name = f"{model_name}__{y}__{k}"
        try:
            with open(f"results/{study_name}.json", "r") as f:
                res = json.load(f)
                try:
                    res = {**res["hyperparameters"],**res["scores"]}
                except:
                    print("Could not load evaluated_scores")
                    continue
                res["model"] = model_name
                res["y"] = y
                res["k"] = k
                scores.append(res)
        except FileNotFoundError:
            print("Could not find file")

score_df = pd.DataFrame.from_dict(scores)
score_df
```

We can see that the model hyperparameters are slightly different across each outer fold.

Now let's tidy it up into a nice table



```{code-cell} ipython3
:tags: [thebe-init]

import numpy as np

cdict = {
    "INCLUDE": "1. Relevance",
    "4 -": "2. Instrument I",
    "8 -": "3. Sector",
}

def mean_std(x):
    m = np.mean(x)
    sd = np.std(x)
    return f"{m:.2} ({sd:.2})"
score_tab = score_df.pivot_table(index="y",columns="model",values="eval_f1", aggfunc=mean_std).reset_index()

score_tab.columns.name = None
score_tab["y"] = score_tab["y"].apply(lambda x: cdict[x])
score_tab = score_tab.rename(columns={"y":"Category"}).fillna("").sort_values("Category", ascending=True)
score_tab
```

## Predictions

Predictions are stored in `results/predictions`, we can merge these with our dataset using the function below

```{code-cell} ipython3
:tags: [thebe-init]

import re

def get_categories(model_name, y_prefix, seen_df, unseen_df, min_samples=15):
    pred = np.load(f"results/predictions/{model_name}__{y_prefix}__predictions.npy")
    ids = np.load(f"results/predictions/{model_name}__{y_prefix}__ids.npy", allow_pickle=True)
    cols = [x for x in seen_df.columns if re.match(y_prefix, x)]
    cols = [x for x in cols if seen_df[x].sum()>min_samples]
    pred_df = pd.DataFrame({"id": ids})
    if y_prefix=='INCLUDE':
        pred_df['INCLUDE']=pred
    else:
        for i, cname in enumerate(cols):
            pred_df[cname]=pred[:,i]
    df = pd.concat([
        unseen_df.merge(pred_df, how='inner'),
        unseen_df.merge(seen_df[['id']+cols], how='inner')
    ]).reset_index(drop=True)
    print(df.shape)
    return df

seen_df = pd.read_feather('data/labels.feather')
unseen_df = pd.read_feather('data/documents.feather')


df = get_categories(model_name, 'INCLUDE', seen_df, unseen_df)
df.head()

```

Documents now have a number between 0 and 1 where a prediction has been made,
and either 0 or 1 if we have a label

```{code-cell} ipython3
:tags: [thebe-init]

df[df['INCLUDE'].isin([0,1])].head()

```

With two more calls to the function, and after filtering only documents we think are relevant, we have our final dataset

```{code-cell} ipython3
:tags: [thebe-init]

df = df[df['INCLUDE']>=0.5]
df = df.dropna(subset=['title','abstract'])

print(df.shape)

df = get_categories(model_name, '4 -', seen_df, df)
df = get_categories(model_name, '8 -', seen_df, df)

df.to_feather('data/final_dataset.feather')

df.head()
```
