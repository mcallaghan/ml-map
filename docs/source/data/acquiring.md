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

# Acquiring Data

Data comes from OpenAlex.

There are more details on how to access OpenAlex data using the API available [here](https://github.com/mcallaghan/NLP-climate-science-tutorial-CCAI/blob/main/A_obtaining_data.ipynb)

In our case, for full flexibility, we download the fortnightly snapshot, and make this searchable using [Solr](https://solr.apache.org/)

In this example tutorial, we make a small sample of our search results available in the data folder.

```{code-cell} ipython3
:tags: [hide-cell, thebe-init]

import os
os.chdir('../../../')
```

These are stored in a [.feather](https://arrow.apache.org/docs/python/feather.html) file

```{code-cell} ipython3
:tags: [thebe-init]

import pandas as pd
df = pd.read_feather('data/documents.feather')
print(df.shape)
df.head()
```
