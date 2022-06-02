# -*- coding: utf-8 -*-
"""The scenario page html layout.

Created on Tue Jul  6 15:23:09 2021

@author: twillia2
"""
from dash import dcc, html

from reView.utils.functions import data_paths
from reView.components import (
    capacity_header,
    above_map_options_div,
    map_div,
    below_map_options_div,
)


PROJECT = str(list(data_paths()["reeds"].glob("*csv"))[0])


layout = html.Div(
    children=[
        # Path Name
        dcc.Location(id="/reeds_page", refresh=False),
        # Constant info block
        html.Div(
            [
                # Project Selection
                html.Div(
                    [
                        html.H4("Project"),
                        dcc.Dropdown(
                            id="project_reeds",
                            options=[
                                # pylint: disable=not-an-iterable
                                {
                                    "label": "Reference Advanced - 95% CO",
                                    "value": PROJECT,
                                }
                            ],
                            value=PROJECT,
                        ),
                    ],
                    className="three columns",
                )
            ]
            + capacity_header(id_prefix="reeds", class_name="three columns"),
            className="row",
            style={"margin-bottom": "35px"},
        ),
        # Year selection
        html.P(
            "Year: ",
            id="year_text",
            className="four columns",
            style={"text-align": "left"},
        ),
        html.Div(
            [dcc.Slider(id="years_reeds", step=2)],
            className="nine columns",
            style={"text-align": "center", "margin-bottom": "55px"},
        ),
        # Above Map Options
        above_map_options_div(id_prefix="reeds", class_name="nine columns"),
        # The map
        map_div(id_prefix="reeds", class_name="nine columns"),
        # Below Map Options
        below_map_options_div(id_prefix="reeds", class_name="nine columns"),
        # Capacity after make_map (avoiding duplicate calls)
        html.Div(id="reeds_mapcap", style={"display": "none"}),
    ]
)
