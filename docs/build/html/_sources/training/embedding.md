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

# Embedding documents

When we fine-tune our model on our dataset, the model will first represent the documents in an embedded space.

We can use these embeddings to make nice visualisations. Running

```
python mlmap/pipeline_embeddings.py -m distilroberta-base
```

will save embeddings in our results folder

```{code-cell} ipython3
:tags: [remove-cell, thebe-init]

import os
os.chdir('../../../')
```

```{code-cell} ipython3
:tags: [thebe-init]

import pandas as pd
import numpy as np

df = pd.read_feather('data/final_dataset.feather')

model_name='distilroberta-base'
embeddings = np.load('results/embeddings.npy')
embeddings.shape
```

## Dimensionality reduction

The embedding space our model uses will have hundreds of dimensions. Because we cannot make nice plots in so many dimensions, we need a way to represent the documents in two dimensions which preserves the differences they have across multiple dimensions. See [link](https://dimensionality-reduction-293e465c2a3443e8941b016d.vercel.app/) for an interactive discussion of dimensionality reduction.

We are going to use [UMAP](https://umap-learn.readthedocs.io/en/latest/) to represent our documents in two dimensions

```{code-cell} ipython3
:tags: [thebe-init]

import umap
reducer = umap.UMAP()
xy = reducer.fit_transform(embeddings)
df['x'] = xy[:, 0]
df['y'] = xy[:, 1]
df.to_feather('data/final_dataset.feather')
xy.shape

```


```{code-cell} ipython3
:tags: [thebe-init]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

ax.scatter(    
    xy[:, 0],
    xy[:, 1],
    s=3,
    alpha=0.2
)
ax.axis("off")

```