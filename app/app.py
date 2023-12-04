from dash import Dash, dash_table, no_update
from components.settings import url_req, url_asset, url_base, url_routes, title
external_stylesheets = [
    'bootstrap.min.css',
]

external_scripts = [
    "jquery-3.5.1.js",
]

app = Dash(
    __name__,
    title=title,
    external_stylesheets=external_stylesheets,
    external_scripts=external_scripts,
    #url_base_pathname=url_base,
    #requests_pathname_prefix=url_req,
    #routes_pathname_prefix=url_routes,
    #assets_url_path=url_asset,
    assets_ignore='fontawesome/js*'
    #assets_url_path='climate-policy-instruments-map/static/'
)

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)