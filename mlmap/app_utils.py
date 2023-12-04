from dash import html, dcc

def doc_template(x):
    res = html.Div([
            html.Div([
                html.Div(x["title"])
            ], className = "card-header d-flex"
            ),
            html.Div([
                html.Div(x['author_name']),
                html.Div(x['institution_name']),
                html.Div(x["publication_year"]),
                html.Div(html.P(
                    x["abstract"],
                    className="card-text text-muted"
                ))
            ], className = "card-body")
        ],
        className="card m-2 p-0 text-start w-100"
    )
    return res
