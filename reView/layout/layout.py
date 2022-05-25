from dash import dcc
from dash import html

from reView.layout.navbar import navbar


# fmt: off
layout = html.Div([
    navbar.NAVBAR,
    dcc.Location(id="url", refresh=False),
    html.Div(id="page_content")
])
