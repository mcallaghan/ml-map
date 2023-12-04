from dash import Dash, dash_table, no_update, html, dcc, ctx
from components.settings import url_req, url_asset, url_base, url_routes, title
from components.template_components import header
from mlmap.app_utils import doc_template
import plotly.graph_objects as go
import json
import pandas as pd
import pyarrow as pa
from dash.dependencies import ClientsideFunction, Input, Output, State, MATCH, ALL
import sqlite3

# How big should the map be
with open('dims.json','r') as f:
    dims = json.load(f)

# The categories we use for filtering are defined here
with open("assets/schema.json", "r") as f:
    schema = json.load(f)

external_stylesheets = [
    'bootstrap.min.css',
]

external_scripts = [
    "assets/jquery-3.5.1.js",
]

app = Dash(
    __name__,
    title=title,
    #external_stylesheets=external_stylesheets,
    external_scripts=external_scripts,
    #url_base_pathname=url_base,
    #requests_pathname_prefix=url_req,
    #routes_pathname_prefix=url_routes,
    #assets_url_path=url_asset,
    #assets_ignore='fontawesome/js*',
    #assets_url_path='static/'
)

server = app.server

##############
## Data
texts = pd.read_feather("assets/texts.feather")
patexts = pa.array(texts["text"])




# We'll initialise the figure with an empty graph
layout = {
    'yaxis': {'visible': False},
    'xaxis': {'visible': False}
}
fig = go.Figure(data=[], layout=layout)

# Now we define the layout for the filter buttons, based on our schema
category_layout = []
categories = []
for category in schema:

    if "colours" in category:
        buttons = [html.Button(x.split('.')[1], id=f"btn_{x.split('.')[1]}", n_clicks=0, className="cbutton clicked m-2 p-1 btn btn-light", style={"backgroundColor":c}) for x, c in zip(category["levels"], category["colours"])]
    else:
        buttons = [html.Button(x.split('.')[1], id=f"btn_{x.split('.')[1]}", n_clicks=0, className="cbutton clicked m-2 p-1 btn") for x in category["levels"]]
    categories += category["levels"]
    category_layout += [
        html.Div([html.H4(category["name"], className='m-2')] + buttons)
    ]

# Now we define a search bar
search = [
    html.H4("Text search", className='m-2'),
    # html.Form([
        html.Div([
            dcc.Input(id="input_search", type="text", className='form-control mb-2', placeholder='search', size='20'),
            html.Button(id='submit', type='submit', children='ok', className='btn btn-primary'),
        ], className='form-group mx-sm-2 mb-2'),

    # ], className='form-inline')
]

# A box for our results
results = [
    html.H4("Results"),
    html.Span(
        f"{texts.shape[0]:,} papers",
        id="n_results"
    ),
    html.Span(" "),
    html.Button('Download', id='btn-download', className='btn btn-primary p-1 m-2'),
    dcc.Download(id='download')
]

app.layout = html.Div(
    html.Div(
        [
            html.Div(
                header,
                className="row"
            ),
            html.Div([
                html.Div(
                    [
                        dcc.Graph(
                            figure=fig,
                            id="scatter",
                            style={
                                "width": f"{70*dims['w_ratio']}vh",
                                "height": "70vh"
                            },
                            className='border rounded',
                            clear_on_unhover=True
                        ),
                        dcc.Tooltip(id="scatter-tooltip", direction='bottom', className='border rounded', style={'width': '200px', 'white-space': 'normal'}),
                        #dcc.Store(id="scatter-data"),
                        dcc.Store(id="search_matches", data = {"filter": False, "ids": [1]})
                    ], className="col"
                ),
                html.Div(
                    [

                        html.Div([html.Button("Load",id="load", className="hidden")]),
                        html.H3('Filters', className='m-2'),
                        html.P([
                            'Click on the buttons below to deselect/select documents with each label ',
                            'Enter a search term into the text search box to search for documents ',
                            'mentioning the term'
                        ], className='m-2')
                    ] +
                    category_layout +
                    [
                        html.Div(search),
                    ],
                className="col bg-light border rounded")

            ], className="row"),
            dcc.Store(id="table_data", data = {}),
            html.Div([

                html.Div(
                    html.Div(results, className='col m-3'),
                    className="row"),
                html.Div(
                    #tab,
                    className="d-flex flex-column flex-shrink-1 item-container",
                    id="itemList"
                )
            ],className="col")
        ],
        className="row g-0"
    ), className="container", id="main"
)

###########################################
## Callbacks

# Download data
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

# Show figure data in results div
@app.callback(
    Output("itemList", "children"),
    Input("table_data", "data"),
    #State("table_data", "data"),
)
def show_texts(d):
    try:
        d = d[:10]
        with sqlite3.connect("data/data.db") as con:
            df = pd.read_sql('SELECT * FROM data WHERE idx IN (%s)' % ("?," * len(d))[:-1], params=d, con=con)

        docs = [doc_template(x) for i, x in df.iterrows()]
        return html.Div(docs, className="d-flex flex-row flex-wrap p-2 overflow-auto border-botton")
    except:
        return

# Filter with text search
@app.callback(
    Output("search_matches", "data"),
    Input("submit", "n_clicks"),
    State("input_search","value"),
    prevent_inital_call=True
)
def filter_texts(n_clicks, search_string):
    if search_string is None or search_string=="":
        return {
            "filter": False,
            "ids": []
        }
    else:
        idx = pa.compute.match_substring_regex(patexts, f"\\b{search_string.lower()}")
        return {
            "filter": True,
            "ids": texts[np.array(idx)].idx.values
        }

# Select items from the scatter plot
@app.callback(
    [
        Output("itemList", "children", allow_duplicate=True)
    ],
    [
        Input("scatter","selectedData")
    ],
    State("table_data", "data"),
    prevent_initial_call=True
)
def select_scatter(selectedData, d):
    print('selected')
    if selectedData:
        if selectedData["points"]:
            idx = [x["pointIndex"] for x in selectedData["points"]]
            d = [d[i] for i in idx][:10]
        else:
            d = d[:10]
    else:
        d = d[:10]
    with sqlite3.connect("data/data.db") as con:
        df = pd.read_sql('SELECT * FROM data WHERE idx IN (%s)' % ("?," * len(d))[:-1], params=d, con=con)
    docs = [doc_template(x) for i, x in df.iterrows()]
    return [html.Div(docs, className="d-flex flex-row flex-wrap p-2 overflow-auto border-botton")]

###########################################
## Clientside callbacks - these functions are defined in javascript and are executed in the browser

# Filter using the buttons
app.clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="button_filter"
    ),

    [
        Output("scatter","figure"),
        Output("n_results", "children"),
        Output("table_data", "data")
    ],
    [
        Input(f"btn_{x.split('.')[1]}", "n_clicks") for x in categories
    ] + [
        Input("search_matches","data"),
        Input('scatter', 'relayoutData'),
        Input('load', 'n_clicks')
    ],
    prevent_inital_call=True
)

# hover over the figure
app.clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="hover"
    ),
    Output("scatter-tooltip", "show"),
    Output("scatter-tooltip", "bbox"),
    Output("scatter-tooltip", "children"),

    Input("scatter", "hoverData"),
    State("table_data", "data"),
    prevent_inital_call=True
)

if __name__ == '__main__':
    app.run_server(debug=True)
