# Building our map with Dash

## Preparing our data

We will start by collecting our data and writing it into a compact format that our app can read

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

##### clientside callbacks
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
