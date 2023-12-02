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

# Fine-tuning a BERT-like model with huggingface

[Huggingface](https://huggingface.co/) is a platform hosting thousands of **pretrained** model, as well as libraries and resources that make it easy for us to **fine-tune them**.

```{code-cell} ipython3
:tags: [remove-cell, thebe-init]

import os
os.chdir('../../../')
```

In the background, huggingface's `transformers` uses either [Pytorch](https://pytorch.org/) or [Tensorflow](https://www.tensorflow.org/). At least one of these has to be installed. In this example, we will use the pytorch backend (see requirements.txt).  

## Datasets

The first step is to get our data (shown below with a very small sample) the huggingface [datasets](https://huggingface.co/docs/datasets/index) format.

```{code-cell} ipython3
:tags: [thebe-init]
from datasets import Dataset
import pandas as pd
df = pd.read_feather('data/labels.feather').sample(32, random_state=2023).reset_index(drop=True)
print(df.title.values)
print(df.INCLUDE.values)
dataset = Dataset.from_dict({"text": df['abstract'], "label": df['INCLUDE']})
dataset
```

## Tokenization

The next step is to **tokenize** our texts. Tokenizers are model specific. In this tutorial we will use [DistilRoberta](https://huggingface.co/distilroberta-base) ([Ro](https://arxiv.org/abs/1907.11692) indicates improvements to the BERT training procedure, [Distil](https://arxiv.org/abs/1910.01108) indicates a smaller, pruned or *distilled* version of the model).

```{code-cell} ipython3
:tags: [thebe-init]

from transformers import AutoTokenizer
model_name = 'distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset
```

We put this into a [function](reference:api): `hf_tokenize_data` function, so that it's simple to create a dataset in the right format. Before using the function, we need to make sure the dataset has a `text` column, and a `labels` column. Usually, we would use the abstract, or the title and the abstract

```{code-cell} ipython3
:tags: [thebe-init]

from mlmap import hf_tokenize_data
df['text'] = df['title'] #+ ' ' + df['abstract']
df['labels'] = df['INCLUDE'].dropna().astype(int)
dataset = hf_tokenize_data(df, model_name)
dataset
```

## Training our model


```{code-cell} ipython3
:tags: [thebe-init]

from transformers import AutoModelForSequenceClassification, Trainer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
trainer = Trainer(model=model, train_dataset=dataset)
# Once this has been instantiated we can apply the train() method
trainer.train()
```

Now we have fine-tuned a model!

## Making predictions with our model

```{code-cell} ipython3
:tags: [thebe-init]

texts = [
  'Designing effective and efficient CO2 mitigation policies in line with Paris Agreement targets',
  'Climate model derived anthropogenic forcing contributions to hurricane intensity '
]
new_df = pd.DataFrame({'text': texts})
dataset = hf_tokenize_data(new_df, model_name)
pred = trainer.predict(dataset)
pred
```

At the moment, these are [logits](). To convert them into probabilities, which are more useful (though these will not be well calibrated), we need an activation function. The [Softmax]() function ensures that probabilities for each class add up to 1 for each document (good for binary classification, when this is represented as a negative and positive class). The [Sigmoid]() function is useful when we have multiple labels that can be true at the same time.

```{code-cell} ipython3
:tags: [thebe-init]

from torch import tensor
from torch.nn import Sigmoid, Softmax
activation = (Softmax())
activation(tensor(pred.predictions))
```

In our codebase, we subclass the `Trainer` class to give it a [predict_proba]() method. This will automatically output probabilities when we make predictions.

## Multilabel predictions

For the instrument type, and the sector, we want to generate a model that predicts what, if any, sectors or instrument types (out of a set of possible values) a document mentions.

To do this, we need to feed a matrix of labels for each instrument type to our model.

Only included documents have instrument types, so lets get a small set of included documents and their instrument types.

```{code-cell} ipython3
:tags: [thebe-init]

import re
df = pd.read_feather('data/labels.feather').query('INCLUDE==1').sample(32, random_state=2023).reset_index(drop=True)
y_prefix = '4 -'
targets = [x for x in df.columns if re.match(f'^y_prefix',x)]
df['labels'] = df[targets].values.astype(int)

```
