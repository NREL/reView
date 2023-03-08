# -*- coding: utf-8 -*-
"""The scenario page html layout.

Created on Tue Jul  6 15:23:09 2021

@author: twillia2
"""
import json

from dash import dcc
from dash import html

from reView.components import (
    capacity_header,
    map_div,
    chart_div,
    time_div,
    REV_PCA_DIV,
    REV_TOPTIONS_DIV,
)
from reView.utils.config import Config


DEFAULT_CONFIG = Config(sorted(Config.projects)[0])
DEFAULT_SIGNAL = {
    "filters": [],
    "mask": "off",
    "path": str(sorted(DEFAULT_CONFIG.all_files)[0]),
    "path2": None,
    "project": DEFAULT_CONFIG.project,
    "recalc": "off",
    "recalc_table": {
        "scenario_a": {
            "fcr": None,
            "capex": None,
            "opex": None,
            "losses": None
        },
        "scenario_b": {
            "fcr": None,
            "capex": None,
            "opex": None,
            "losses": None
        }
    },
    "added_scenarios": [],
    "regions": None,
    "diff_units": "original",
    "states": None,
    "x": "capacity",
    "y": "mean_lcoe",
}

layout = html.Div(
    className="eleven columns",
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
        dcc.Location(id="/scenario_page", refresh=False),

        # Constant info block and options
        html.Div(
            capacity_header(id_prefix="rev", class_name="four columns"),
            style={"margin-left": "300px"}
        ),
        REV_TOPTIONS_DIV,
        REV_PCA_DIV,

        # Map and chart
        html.Div(
            children=[
                # The map
                map_div(id_prefix="rev", class_name="six columns"),

                # The chart
                chart_div(id_prefix="rev", class_name="six columns"),

                # The timeseries if available
                html.Div(
                    id="timeseries",
                    children=time_div(
                        id_prefix="rev",
                        class_name="twelve columns"
                    )        
                ),
            ],
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
        html.Div(id="map_signal", children=json.dumps(DEFAULT_SIGNAL),
                 style={"display": "none"}),

        # This table of recalc parameters
        html.Div(
            id="recalc_table_store",
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

        # Download
        dcc.Download(id="download_chart"),
        dcc.Download(id="download_map"),
        html.Div(id="download_info_chart", style={"display": "none"}),
    ],
)
