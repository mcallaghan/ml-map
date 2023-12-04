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
  execution_mode: force
---

# Building our map with Dash

## Preparing our data

We will start by collecting our data and writing it into a compact format that our app can read


```{code-cell} ipython3
:tags: [remove-cell, thebe-init]

import os
os.chdir('../../../')
```

```{code-cell} ipython3
:tags: [thebe-init]

import pandas as pd
import numpy as np
import sqlite3
import gzip
import json

df = pd.read_feather('data/final_dataset.feather')
df = df.fillna(0)
df['idx'] = df.index

# We need a single sector to colour code the dots, let's get the maximum
sectors = [x for x in df.columns if "8 -" in x]
df['sector'] = df[sectors].apply(lambda x: sectors[np.argmax(x)], axis=1)

# We'll write it out in json format in 5 chunks
chunk_size = df.shape[0] // 5
for i, group in df.groupby(np.arange(len(df))//chunk_size):
    d = {x: list(group[x]) for x in df.columns}
    json_str = json.dumps(d)
    json_bytes = json_str.encode('utf-8')   
    with gzip.open(f'app/assets/data_{i}.json', 'w') as f:
        f.write(json_bytes)

# We'll also write out a database
with sqlite3.connect("app/data/data.db") as con:
    cursor = con.cursor()
    cursor.execute("DROP TABLE IF EXISTS data ")
    df.to_sql('data',con)

# And we'll write a table of just the texts
df['text'] = df['title'] + ' ' + df['abstract']
df[['idx','text']].to_feather('app/assets/texts.feather')

```

## Dash app

### app.py

`app.py` Describes the appearance of your application and how it can be interacted with

#### Layout

With `app.layout`, we define the components that make up app in a nested structure.
The customisable parts are defined in `components/*``

#### Callbacks

Callbacks are functions that define what to do when the page is interacted with.
We need to define what triggers the callback `Input`, what stored information we want to use
`State`, and what on the page we want to change `Output`.

The callback below is triggered by clicking on the download button. It collects the ids stored in the
`table_data` state, and retrieves the corresponding records from our database, constructs a csv file from
that, and passes this to be downloaded

```
@app.callback(
    Output('download','data'),
    Input('btn-download','n_clicks'),
    State("table_data", "data"),
    prevent_initial_call=True
)
def download_data(n_clicks, d):
    with sqlite3.connect("data/data.db") as con:
        q = f'SELECT * FROM data WHERE idx IN ({",".join([str(x) for x in d])})'
        download_df = pd.read_sql(q, con=con)
    return dict(content=download_df.to_csv(index=False), filename="climate_policy_papers.csv")
```

##### Clientside callbacks
clientside callbacks are those written in javascript code, and which are executed on the browser.
This is useful to prevent large amounts of data being transferred between the client and the browser during interactions.

They are defined in `assets/index.js`, however this file is compiled from source/index.js

Compiling javscript files allows you to make use of external javascript libraries, which are managed by npm

To compile this file you will need to install npm packages

```
npm i
```
And assemble the file
```
npx browserify -p esmify src/index.js -o assets/index.js
```

### Running the app

Finally, we can run our app with

`python app.py`
