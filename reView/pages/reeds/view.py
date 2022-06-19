# -*- coding: utf-8 -*-
"""The scenario page html layout.

Created on Tue Jul  6 15:23:09 2021

@author: twillia2
"""
from dash import dcc, html

from reView.utils.functions import data_paths
from reView.components import capacity_header, map_div


PROJECT = str(list(data_paths()["reeds"].glob("*csv"))[0])


layout = html.Div(
    className="twelve columns",
    style={
        "margin-top": "100px",
        "margin-bottom": "100px",
        "margin-right": "3%",
        "margin-left": "3%",
        "backgroundColor": "white",
        "text-align": "center",
    },
    children=[
        # Path Name
        dcc.Location(id="/reeds_page", refresh=False),

        # Constant info block
        html.Div(
            [
                # Project Selection
                html.Div(
                    [
                        html.H4("Output #1"),
                        dcc.Dropdown(
                            id="project_reeds_1",
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
                    className="four columns",
                    style={"margin-right": "100px"},
                ),
                html.Div(
                    [
                        html.H4("Output #2"),
                        dcc.Dropdown(
                            id="project_reeds_2",
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
                    className="four columns",
                    style={"margin-right": "100px"},
                ),
            ],
            className="twelve columns",
            style={"margin-bottom": "35px", "margin-left": "150px"},
        ),

        # Year selection
        html.H4(
            id="year_text",
            className="four columns",
            style={"text-align": "left", "margin-left": "200px"},
        ),
        html.Div(
            children=[
                dcc.Slider(
                    id="years_reeds",
                    step=1,
                    value=None
                ),
            ],
            className="nine columns",
            style={"text-align": "center", "margin-bottom": "50px",
                   "margin-left": "200px"},
        ),

        # The maps
        html.Div(
            className="five columns",
            children=[
                capacity_header(
                    id_prefix="reeds_1",
                    class_name="five columns",
                    cap_title="Capacity:",
                    count_title="Sites:",
                    style={"margin-left": "100px"},
                    small=True
                ),
                map_div(id_prefix="reeds_1", class_name="twelve columns"),
           ]
        ),
        html.Div(
            className="five columns",
            children=[
                capacity_header(
                    id_prefix="reeds_2",
                    class_name="five columns",
                    cap_title="Capacity:",
                    count_title="Sites:",
                    style={"margin-left": "100px"},
                    small=True
                ),
                map_div(id_prefix="reeds_2", class_name="twelve columns"),
           ]
        ),

        # Capacity after make_map (avoiding duplicate calls)
        html.Div(id="reeds_1_mapcap", style={"display": "none"}),
        html.Div(id="reeds_2_mapcap", style={"display": "none"}),

        # Store callback number to keep dropdowns different
        html.Div(id="call_1", children="0", style={"display": "none"}),
        html.Div(id="call_2", children="1", style={"display": "none"}),
    ],
)
