from components.settings import title
from dash import html, dcc

header = html.Div([
    html.H1(title, className="p-2"),
    html.P([
        "[author]",
        #DI(icon='ion:logo-github')
    ], className="text-muted"),
    html.P([
        "This interactive website accompanies the paper [x],",
        " which uses machine learning to identify and classify the literature on [y].",
        " You can explore this literature in the map below, where each paper is represented by a dot, and ",
        "papers which are linguistically similar are placed close together on the plot. ",
        "Hovering over the map will show the titles of the papers."
    ]),
    html.P([
        "You can select papers by clicking and dragging on the map to zoom in on an area. ",
        "Or you can choose a different type of selection method using the icons in the top left. "
        "Once you have selected documents, a sample of these will be shown in the box below. ",
        "You will also have the opportunity to download the complete selection of documents, including ",
        "the machine-learning generated labels."
    ])
],className='row')
