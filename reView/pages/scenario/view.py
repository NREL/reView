# -*- coding: utf-8 -*-
"""The scenario page html layout.

Created on Tue Jul  6 15:23:09 2021

@author: twillia2
"""
import json

from dash import dcc
from dash import html

from reView.layout.styles import (
    BUTTON_STYLES,
    TAB_STYLE,
    TAB_BOTTOM_SELECTED_STYLE,
    TABLET_STYLE,
    TABLET_STYLE_CLOSED,
)
from reView.utils.classes import DiffUnitOptions
from reView.environment.settings import IS_DEV_ENV
from reView.utils.config import Config
from reView.components import (
    capacity_header,
    map_div,
    chart_div,
    REV_PCA_DIV,
    REV_TOPTIONS_DIV,
)


layout = html.Div(
    className="eleven columns",
    style={
        "margin-top": "100px",
        "margin-bottom": "100px",
        "margin-right": "3%",
        "margin-left": "3%",
        "backgroundColor": "white",
        "text-align": "center"
    },
    children=[
        # Path Name
        dcc.Location(id="/scenario_page", refresh=False),

        # Constant info block
        html.Div(
            [
                # Project Selection
                html.Div(
                    [
                        html.H4("Project"),
                        dcc.Dropdown(
                            id="project",
                            options=[
                                # pylint: disable=not-an-iterable
                                {"label": project, "value": project}
                                for project in Config.sorted_projects
                            ],
                        ),
                    ],
                    className="three columns",
                ),
                capacity_header(
                    id_prefix="rev",
                    class_name="four columns"
                ),
            ],
            className="ten columns",
            style={
                "margin-top": "30px",
                "margin-bottom": "35px",
                "margin-left": "150px"
            },
        ),

        REV_TOPTIONS_DIV,

        REV_PCA_DIV,

        # Map and chart
        html.Div(
            children=[
                # The map
                map_div(
                    id_prefix="rev",
                    class_name="six columns"
                ),
                # The chart
                chart_div(
                    id_prefix="rev",
                    class_name="six columns"
                    
                ),            
            ]
        ),

        # To store option names for the map title
        html.Div(id="chosen_map_options", style={"display": "none"}),

        # To store option names for the chart title
        html.Div(id="chosen_chart_options", style={"display": "none"}),

        # For storing the data frame path and triggering updates
        html.Div(id="map_data_path", style={"display": "none"}),

        # For storing the signal need for the set of chart data frames
        html.Div(id="chart_data_signal", style={"display": "none"}),

        # Interim way to share data between map and chart
        html.Div(id="map_signal", style={"display": "none"}),

        # This table of recalc parameters
        html.Div(
            id="recalc_table",
            children=json.dumps(
                {
                    "scenario_a": {
                        "fcr": None,
                        "capex": None,
                        "opex": None,
                        "losses": None,
                    },
                    "scenario_b": {
                        "fcr": None,
                        "capex": None,
                        "opex": None,
                        "losses": None,
                    },
                }
            ),
            style={"display": "none"},
        ),

        # Capacity after make_map (avoiding duplicate calls)
        html.Div(id="rev_mapcap", style={"display": "none"}),

        # Filter list after being pieced together
        html.Div(id="filter_store", style={"display": "none"}),
    ]
)
