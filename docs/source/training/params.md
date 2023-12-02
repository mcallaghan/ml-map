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

# Optimizing hyperparameters


```{code-cell} ipython3
:tags: [remove-cell, thebe-init]

import os
os.chdir('../../../')
```

The model we have just trained works OK, but there are several choices we can make (beyond simply using the default parameters) about how the model should go about training. These choices, or **hyperparameters** will affect the performance of our model, in ways that we cannot always predict *a priori*. It is however likely that the default parameters are not the best ones.

## Choosing parameters with huggingface

We can set the parameters with a [TrainingArguments](https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/trainer#transformers.TrainingArguments) object, which we then pass to our Trainer.

The orignal BERT paper recommends we explore the following hyperparameter space

```{code-cell} ipython3
:tags: [thebe-init]

params = {
  "per_device_train_batch_size": [16, 32],
  "learning_rate": [5e-5, 3e-5, 2e-5],
  "num_train_epochs": [2,3,4]
}
```

We can turn this into a list of unique combinations using itertools

```{code-cell} ipython3
:tags: [thebe-init]

import itertools
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
param_list = list(product_dict(**params))
print(len(param_list))
```

```{code-cell} ipython3
:tags: [thebe-init, remove-cell]

from myst_nb import glue
glue('n_params', len(param_list))
```

There are {glue:}`n_params` unique combinations of parameters in there. We'll first explore how we can test out one set.

One thing we'll need to do is to separate a test set of documents from our training set. Our training procedure will not see these documents, and we'll see how well our model does at predicting the right values for documents it has not seen before.

```{code-cell} ipython3
:tags: [thebe-init]

from datasets import Dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from mlmap import hf_tokenize_data, CustomTrainer
df = pd.read_feather('data/labels.feather').sample(128, random_state=2023).reset_index(drop=True)
df['text'] = df['title'] + ' ' + df['abstract']
y_prefix = 'INCLUDE'
targets = [x for x in df.columns if re.match(f'^{y_prefix}',x)]
if len(targets)==1:
    df['labels'] = df[targets[0]]
    binary=True
else:
    df['labels'] = df[targets]
    binary=False

model_name = 'distilroberta-base'
dataset = hf_tokenize_data(df, model_name)
train_idx, test_idx = train_test_split(df.index)
train_data = dataset.select(train_idx)
test_data = dataset.select(test_idx)
```

Now we have split our data up, we want to train a model with a given set of parameters on our training data, and test it on our testing data

```{code-cell} ipython3
:tags: [thebe-init]

from transformers import Trainer, AutoModelForSequenceClassification, TrainingArguments
p = param_list[0]
training_args = TrainingArguments(
    output_dir='./results',
    save_steps=1e9,
    optim='adamw_torch'
)
for k, v in p.items():
    setattr(training_args,k,v)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
trainer = CustomTrainer(model, train_dataset=train_data, args=training_args)
trainer.train()
```

Now we can see how well this work on our test dataset


```{code-cell} ipython3
:tags: [thebe-init]

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
pred = trainer.predict_proba(test_data, binary=binary)

y_true = df.loc[test_idx,'labels']
glue('f1_p0', f1_score(y_true, pred.round()))

ConfusionMatrixDisplay.from_predictions(
    y_true,
    pred.round(),
    cmap='Blues'
)


```

This model achieved and f1 score of {glue:}`f1_p0`.

## Class weighting

The optimization procedure of our model penalizes mistaken classifications for all classes equally. Where we have unbalanced classes, this might mean our model gets good at predicting one common class at the expense of another less common class. This would not be ideal behaviour.

We can penalize infrequent classes more heavily by calculating weights as follows


```{code-cell} ipython3
:tags: [thebe-init]

from torch import tensor, cuda, device
device = "cuda:0" if cuda.is_available() else "cpu"
weights = tensor(df.shape[0] / df[targets].sum(axis=0))
weights = weights.to(device)
weights
```

We can subclass TrainingArguments 

```{code-cell} ipython3
:tags: [thebe-init]

from mlmap import CustomTrainingArguments

training_args = CustomTrainingArguments(
    output_dir='./results',
    save_steps=1e9,
    optim='adamw_torch'
)
training_args.class_weights = weights
training_args.use_class_weights = True
for k, v in p.items():
    setattr(training_args,k,v)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
trainer = CustomTrainer(model, train_dataset=train_data, args=training_args)
trainer.train()

```

```{code-cell} ipython3
:tags: [thebe-init]

pred = trainer.predict_proba(test_data, binary=binary)

y_true = df.loc[test_idx,'labels']
glue('f1_p0c', f1_score(y_true, pred.round()))

ConfusionMatrixDisplay.from_predictions(
    y_true,
    pred.round(),
    cmap='Blues'
)


```

This model achieved and f1 score of {glue:}`f1_p0c`.


```{code-cell} ipython3
:tags: [thebe-init]


```
