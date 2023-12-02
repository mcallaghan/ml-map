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

# Labelling Data

Labelling data is laborious. We spent months reading abstracts by hand.

[pic]

Each document was coded by hand by at least two independent coders. All disagreements

```{code-cell} ipython3
:tags: [remove-cell, thebe-init]

import os
os.chdir('../../../')
```

A sample of documents we labelled are also included (for demonstration purposes here with a subset of the most common labels).

```{code-cell} ipython3
:tags: [thebe-init]

import pandas as pd
labels = pd.read_feather('data/labels.feather')
print(labels.shape)
labels.head()
```

## Inclusion

`INCLUDE` is a binary label that takes the value of 1 when a document was included (meaning that it deals with policy instruments of some sort)

```{code-cell} ipython3
:tags: [thebe-init]

import matplotlib.pyplot as plt
labels.groupby('INCLUDE')['id'].count().plot.bar()
```

## Policy Instrument Type

Policy instrument types are denoted by columns beginning with the prefix `4 -`. Taken together, they can be seen as a multilabel task (each document can be zero or more policy instruments)

```{code-cell} ipython3
:tags: [thebe-init]

instruments = [x for x in labels.columns if "4 -" in x]
labels[instruments].sum().plot.bar()
```

## Sector

Sectors are denoted by columns beginning with the prefix `4 -`. They can also be seen as a multilabel task (each document can be zero or more sectors)

```{code-cell} ipython3
:tags: [thebe-init]

sectors = [x for x in labels.columns if "8 -" in x]
labels[sectors].sum().plot.bar()
```

Documents that are relevant usually mention 1 or more specific instrument types in 1 or more sectors (cross-sectoral refers to instruments that simply talk about reducing emissions in general)

```{code-cell} ipython3
:tags: [thebe-init]

import numpy as np
import seaborn as sns
m = np.zeros((len(sectors),len(instruments)))
for i, sec in enumerate(sectors):
  for j, inst in enumerate(instruments):
    m[i,j] = labels[(labels[sec]==1) & (labels[inst]==1)].shape[0]
sns.heatmap(
  m,
  xticklabels=instruments,
  yticklabels=sectors,
  cmap='Blues',
  annot=True
)
plt.show()
```
