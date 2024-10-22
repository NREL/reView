# -*- coding: utf-8 -*-
"""reView Layout generator
    Creates parent DIV for containing page layout
"""
from dash import dcc
from dash import html

from reView.layout import navbar


# fmt: off
def get_layout():
    """Get the application layout."""
    layout = html.Div([
        navbar.NAVBAR,
        dcc.Location(id="url", refresh=True),
        html.Div(id="page_content"),
        navbar.SIDE_BUTTON,
    ])
    return layout