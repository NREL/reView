# -*- coding: utf-8 -*-
"""The scenario page html layout.

Created on Tue Jul  6 15:23:09 2021

@author: twillia2
"""
import json

from dash import dcc
from dash import html

import plotly.graph_objects as go

from reView.layout.styles import (
    BOTTOM_DIV_STYLE,
    BUTTON_STYLES,
    TAB_STYLE,
    TAB_BOTTOM_SELECTED_STYLE,
    TABLET_STYLE,
    TABLET_STYLE_CLOSED,
)
from reView.layout.options import CHART_OPTIONS, REGION_OPTIONS
from reView.utils.constants import DEFAULT_POINT_SIZE
from reView.utils.classes import DiffUnitOptions
from reView.environment.settings import IS_DEV_ENV
from reView.utils.config import Config
from reView.components import (
    above_map_options_div,
    map_div,
    below_map_options_div,
)


layout = html.Div(
    # className="eleven columns",
    style={"margin-left": "3%", "margin-right": "3%"},
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

                # Print total capacity after all the filters are applied
                html.Div(
                    [
                        html.H5("Remaining Generation Capacity: "),
                        dcc.Loading(
                            children=[
                                html.H1(id="capacity_print", children=""),
                            ],
                            type="circle",
                        ),
                    ],
                    className="three columns",
                ),

                # Print total capacity after all the filters are applied
                html.Div(
                    [
                        html.H5("Number of Sites: "),
                        dcc.Loading(
                            children=[
                                html.H1(id="site_print", children=""),
                            ],
                            type="circle",
                        ),
                    ],
                    className="three columns",
                ),
            ],
            className="row",
            style={"margin-bottom": "35px"},
        ),

        # Toggle Options Top
        html.Div(
            [
                html.Button(
                    id="toggle_options",
                    children="Options: Off",
                    n_clicks=0,
                    type="button",
                    title=("Click to display options"),
                    style=BUTTON_STYLES["off"],
                    className="two columns",
                ),
            ],
            className="row",
            style={
                "margin-left": "50px",
                "margin-right": "1px",
                "margin-bottom": "1px",
            },
        ),
        html.Hr(
            className="row",
            style={
                "width": "92%",
                "margin-left": "53px",
                "margin-right": "10px",
                "margin-bottom": "-1px",
                "margin-top": "-1px",
                "border-bottom": "2px solid #fccd34",
                "border-top": "3px solid #1663b5",
            },
        ),

        # Scen selection tabs - Tabs for selection options
        html.Div(
            id="options_div",
            # className="ten columns",
            style={
                "width": "92%",
                "text-align": "center",
                "margin-left": "53px",
                "margin-right": "10px",
                "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)",
            },
            children=[
                dcc.Tabs(
                    id="scenario_selection_tabs",
                    value="0",
                    children=[
                        dcc.Tab(label="Default Scen", value="0"),
                        dcc.Tab(
                            label="Lowest Scen",
                            value="1",
                        ),
                        dcc.Tab(
                            label="PCA" if IS_DEV_ENV else "Under construction",
                            value="2",
                            disabled=not IS_DEV_ENV,
                        ),
                    ],
                    style={"display": "none"},
                ),

                # Data Options
                html.Div(
                    [
                        # First Scenario
                        html.Div(
                            [
                                html.H5("Scenario A"),
                                html.Div(
                                    id="scenario_a_options",
                                ),
                            ],
                            className="three columns",
                            style={"margin-left": "50px"},
                        ),
                        # Second Scenario
                        html.Div(
                            id="scenario_b_div",
                            children=[
                                html.Div(
                                    [
                                        html.H5("Scenario B"),
                                        html.Div(
                                            id="scenario_b_options",
                                        ),
                                    ],
                                    className="three columns",
                                )
                            ],
                            style={"margin-left": "50px"},
                        ),
                        # Variable options
                        html.Div(
                            [
                                html.H5("Variable"),
                                dcc.Dropdown(
                                    id="variable",
                                    options=[
                                        {"label": "Capacity", "value": "capacity"}
                                    ],
                                    value="capacity",
                                ),
                            ],
                            className="two columns",
                        ),
                        # Show difference map
                        html.Div(
                            [
                                html.H5("Scenario B Difference"),
                                dcc.Tabs(
                                    id="difference",
                                    value="off",
                                    style=TAB_STYLE,
                                    children=[
                                        dcc.Tab(
                                            value="on",
                                            label="On",
                                            style=TABLET_STYLE,
                                            selected_style=TABLET_STYLE_CLOSED,
                                        ),
                                        dcc.Tab(
                                            value="off",
                                            label="Off",
                                            style=TABLET_STYLE,
                                            selected_style=TABLET_STYLE_CLOSED,
                                        ),
                                    ],
                                ),
                                dcc.Tabs(
                                    id="difference_units",
                                    value=str(DiffUnitOptions.PERCENTAGE),
                                    style=TAB_STYLE,
                                    children=[
                                        dcc.Tab(
                                            value=str(DiffUnitOptions.PERCENTAGE),
                                            label="Percentage",
                                            style=TABLET_STYLE,
                                            selected_style=TAB_BOTTOM_SELECTED_STYLE,
                                        ),
                                        dcc.Tab(
                                            value=str(DiffUnitOptions.ORIGINAL),
                                            label="Original Units",
                                            style=TABLET_STYLE,
                                            selected_style=TAB_BOTTOM_SELECTED_STYLE,
                                        ),
                                    ],
                                ),
                                dcc.Tabs(
                                    id="mask",
                                    value="off",
                                    style=TAB_STYLE,
                                    children=[
                                        dcc.Tab(
                                            value="off",
                                            label="No Mask",
                                            style=TABLET_STYLE,
                                            selected_style=TAB_BOTTOM_SELECTED_STYLE,
                                        ),
                                        dcc.Tab(
                                            value="on",
                                            label="Scenario B Mask",
                                            style=TABLET_STYLE,
                                            selected_style=TAB_BOTTOM_SELECTED_STYLE,
                                        ),
                                    ],
                                ),
                                html.Hr(),
                            ],
                            className="two columns",
                        ),

                        # Add in a map function option (demand meetiing)
                        html.Div(
                            id="map_function_div",
                            className="two columns",
                            children=[
                                html.H5("Mapping Function"),
                                dcc.Dropdown(
                                    id="map_function",
                                    options=[
                                        {"label": "None", "value": "None"},
                                        {
                                            "label": "Single Load Demand",
                                            "value": "demand",
                                        },
                                        {
                                            "label": "Meet Demand",
                                            "value": "meet_demand",
                                        },
                                    ],
                                    value="None",
                                ),
                            ],
                        ),

                        # LCOE Recalc
                        html.Div(
                            [
                                html.H5(
                                    "Recalculate With New Costs*",
                                    title=(
                                        "Recalculating will not re-sort "
                                        "transmission connections so there will be "
                                        "some error with Transmission Capital "
                                        "Costs, LCOT, and Total LCOE."
                                    ),
                                ),
                                dcc.Tabs(
                                    id="recalc_tab",
                                    value="off",
                                    style=TAB_STYLE,
                                    children=[
                                        dcc.Tab(
                                            value="on",
                                            label="On",
                                            style=TABLET_STYLE,
                                            selected_style=TABLET_STYLE_CLOSED,
                                        ),
                                        dcc.Tab(
                                            value="off",
                                            label="Off",
                                            style=TABLET_STYLE,
                                            selected_style=TABLET_STYLE_CLOSED,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="recalc_tab_options",
                                    children=[
                                        dcc.Tabs(
                                            id="recalc_scenario",
                                            value="scenario_a",
                                            style=TAB_STYLE,
                                            children=[
                                                dcc.Tab(
                                                    value="scenario_a",
                                                    label="Scenario A",
                                                    style=TABLET_STYLE,
                                                    selected_style=TABLET_STYLE_CLOSED,
                                                ),
                                                dcc.Tab(
                                                    value="scenario_b",
                                                    label="Scenario B",
                                                    style=TABLET_STYLE,
                                                    selected_style=TABLET_STYLE_CLOSED,
                                                ),
                                            ],
                                        ),

                                        # Long table of scenario A recalc parameters
                                        html.Div(
                                            id="recalc_a_options",
                                            children=[
                                                # FCR A
                                                html.Div(
                                                    [
                                                        html.P(
                                                            "FCR % (A): ",
                                                            className="three columns",
                                                            style={"height": "60%"},
                                                        ),
                                                        dcc.Input(
                                                            id="fcr1",
                                                            type="number",
                                                            className="nine columns",
                                                            style={"height": "60%"},
                                                            value=None,
                                                        ),
                                                    ],
                                                    className="row",
                                                ),
                                                # CAPEX A
                                                html.Div(
                                                    [
                                                        html.P(
                                                            "CAPEX $/KW (A): ",
                                                            className="three columns",
                                                            style={"height": "60%"},
                                                        ),
                                                        dcc.Input(
                                                            id="capex1",
                                                            type="number",
                                                            className="nine columns",
                                                            style={"height": "60%"},
                                                        ),
                                                    ],
                                                    className="row",
                                                ),
                                                # OPEX A
                                                html.Div(
                                                    [
                                                        html.P(
                                                            "OPEX $/KW (A): ",
                                                            className="three columns",
                                                            style={"height": "60%"},
                                                        ),
                                                        dcc.Input(
                                                            id="opex1",
                                                            type="number",
                                                            className="nine columns",
                                                            style={"height": "60%"},
                                                        ),
                                                    ],
                                                    className="row",
                                                ),
                                                # Losses A
                                                html.Div(
                                                    [
                                                        html.P(
                                                            "Losses % (A): ",
                                                            className="three columns",
                                                            style={"height": "60%"},
                                                        ),
                                                        dcc.Input(
                                                            id="losses1",
                                                            type="number",
                                                            className="nine columns",
                                                            style={"height": "60%"},
                                                        ),
                                                    ],
                                                    className="row",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="recalc_b_options",
                                            children=[
                                                # FCR B
                                                html.Div(
                                                    [
                                                        html.P(
                                                            "FCR % (B): ",
                                                            className="three columns",
                                                            style={"height": "60%"},
                                                        ),
                                                        dcc.Input(
                                                            id="fcr2",
                                                            type="number",
                                                            className="nine columns",
                                                            style={"height": "60%"},
                                                        ),
                                                    ],
                                                    className="row",
                                                ),
                                                # CAPEX B
                                                html.Div(
                                                    [
                                                        html.P(
                                                            "CAPEX $/KW (B): ",
                                                            className="three columns",
                                                            style={"height": "60%"},
                                                        ),
                                                        dcc.Input(
                                                            id="capex2",
                                                            type="number",
                                                            className="nine columns",
                                                            style={"height": "60%"},
                                                        ),
                                                    ],
                                                    className="row",
                                                ),
                                                # OPEX B
                                                html.Div(
                                                    [
                                                        html.P(
                                                            "OPEX $/KW (B): ",
                                                            className="three columns",
                                                            style={"height": "60%"},
                                                        ),
                                                        dcc.Input(
                                                            id="opex2",
                                                            type="number",
                                                            className="nine columns",
                                                            style={"height": "60%"},
                                                        ),
                                                    ],
                                                    className="row",
                                                ),
                                                # Losses B
                                                html.Div(
                                                    [
                                                        html.P(
                                                            "Losses % (B): ",
                                                            className="three columns",
                                                            style={"height": "60%"},
                                                        ),
                                                        dcc.Input(
                                                            id="losses2",
                                                            type="number",
                                                            className="nine columns",
                                                            style={"height": "60%"},
                                                        ),
                                                    ],
                                                    className="row",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Hr(),
                            ],
                            id="recalculate_with_new_costs",
                            className="four columns",
                        ),

                        # Filters
                        html.Div(
                            [
                                html.H5(
                                    "Filters",
                                    title=(
                                        "Filter map and charts with variable "
                                        "value thresholds. Enter <variable> "
                                        "<operator> <value> (working on a "
                                        "more intuitive way)"
                                    ),
                                    id="filter_title",
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="filter_variables_1",
                                            placeholder="Choose variable.",
                                            className="six columns",
                                            style={"margin-right": -1},
                                        ),
                                        dcc.Input(
                                            id="filter_1",
                                            placeholder="Filter 1",
                                            className="six columns",
                                            style={
                                                "background-color": "#f9f9f9",
                                                "margin-left": -1,
                                            },
                                        ),
                                    ],
                                    style={"width": "100%"},
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="filter_variables_2",
                                            placeholder="Choose variable.",
                                            className="six columns",
                                            style={
                                                "margin-right": -1,
                                            },
                                        ),
                                        dcc.Input(
                                            id="filter_2",
                                            placeholder="Filter 2",
                                            className="six columns",
                                            style={
                                                "background-color": "#f9f9f9",
                                                "margin-left": -1,
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="filter_variables_3",
                                            placeholder="Choose variable.",
                                            className="six columns",
                                            style={
                                                "margin-right": -1,
                                            },
                                        ),
                                        dcc.Input(
                                            id="filter_3",
                                            placeholder="Filter 3",
                                            className="six columns",
                                            style={
                                                "background-color": "#f9f9f9",
                                                "margin-left": -1,
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="filter_variables_4",
                                            placeholder="Choose variable.",
                                            className="six columns",
                                            style={
                                                "margin-right": -1,
                                            },
                                        ),
                                        dcc.Input(
                                            id="filter_4",
                                            placeholder="Filter 4",
                                            className="six columns",
                                            style={
                                                "background-color": "#f9f9f9",
                                                "margin-left": -1,
                                            },
                                        ),
                                    ],
                                    className="twelve columns",
                                ),
                            ],
                            className="four columns",
                        ),
                    ],
                    id="options",
                    className="row",
                    style={"display": "none"},
                ),
                html.Div(
                    [
                        # First Scenario
                        html.Div(
                            [
                                html.H5("Scenario"),
                                html.Div(
                                    id="minimizing_scenario_options",
                                ),
                            ],
                            className="three columns",
                            style={"margin-left": "50px"},
                        ),
                        # Variable options
                        html.Div(
                            [
                                html.H5("Variable"),
                                dcc.Dropdown(
                                    id="minimizing_variable",
                                    options=[{"label": "None", "value": "None"}],
                                    value="None",
                                ),
                            ],
                            className="two columns",
                        ),
                        # Target options
                        html.Div(
                            [
                                html.H5("Minimization Target"),
                                dcc.Dropdown(
                                    id="minimizing_target",
                                    options=[{"label": "None", "value": "None"}],
                                    value="None",
                                ),
                            ],
                            className="two columns",
                        ),
                        # Plot options
                        html.Div(
                            [
                                html.H5("Plot Value"),
                                dcc.Dropdown(
                                    id="minimizing_plot_value",
                                    options=[
                                        {"label": "Variable", "value": "Variable"},
                                    ],
                                    value="Variable",
                                ),
                            ],
                            className="two columns",
                        ),
                    ],
                    id="minimizing_scenarios",
                    className="row",
                    style={"display": "none"},
                ),
                html.Div(
                    [
                        # Both PCA plot
                        html.Div(
                            [
                                # The PCA plot
                                dcc.Graph(
                                    id="pca_plot_1",
                                    style={"height": 500, "width": 800},
                                    className="row",
                                    # style={"margin-left": "50px"},
                                    config={
                                        "showSendToCloud": True,
                                        "plotlyServerURL": "https://chart-studio.plotly.com",
                                        "toImageButtonOptions": {
                                            "width": 500,
                                            "height": 500,
                                            "filename": "custom_pca_plot",
                                        },
                                    },
                                ),
                                # The second PCA plot
                                dcc.Graph(
                                    id="pca_plot_2",
                                    style={"height": 500, "width": 800},
                                    className="row",
                                    # style={"margin-left": "50px"},
                                    config={
                                        "showSendToCloud": True,
                                        "plotlyServerURL": "https://chart-studio.plotly.com",
                                        "toImageButtonOptions": {
                                            "width": 500,
                                            "height": 500,
                                            "filename": "custom_pca_plot",
                                        },
                                    },
                                ),
                            ],
                            className="two rows",
                        ),
                        # Below PCA plot Options
                        html.Div(
                            [
                                # Left options
                                html.Div(
                                    [
                                        html.P(
                                            "Top Min: ",
                                            # style={
                                            #     "margin-left": 5,
                                            #     "margin-top": 7,
                                            # },
                                            className="column",
                                        ),
                                        dcc.Input(
                                            id="pca1_color_min",
                                            type="number",
                                            debounce=False,
                                            className="column",
                                            # style={
                                            #     "margin-left": "-1px",
                                            #     "width": "15%",
                                            # },
                                        ),
                                        html.P(
                                            "Top Max: ",
                                            style={"margin-top": 7},
                                            className="column",
                                        ),
                                        dcc.Input(
                                            id="pca1_color_max",
                                            debounce=False,
                                            type="number",
                                            className="column",
                                            # style={
                                            #     "margin-left": "-1px",
                                            #     "width": "15%",
                                            # },
                                        ),
                                        html.P(
                                            "Bottom Min: ",
                                            # style={
                                            #     "margin-left": 5,
                                            #     "margin-top": 7,
                                            # },
                                            className="column",
                                        ),
                                        dcc.Input(
                                            id="pca2_color_min",
                                            type="number",
                                            debounce=False,
                                            className="column",
                                            # style={
                                            #     "margin-left": "-1px",
                                            #     "width": "15%",
                                            # },
                                        ),
                                        html.P(
                                            "Bottom Max: ",
                                            style={"margin-top": 7},
                                            className="column",
                                        ),
                                        dcc.Input(
                                            id="pca2_color_max",
                                            debounce=False,
                                            type="number",
                                            className="column",
                                            # style={
                                            #     "margin-left": "-1px",
                                            #     "width": "15%",
                                            # },
                                        ),
                                    ],
                                    className="eight columns",
                                    # style=BOTTOM_DIV_STYLE,
                                ),
                            ]
                        ),
                        # PCA Plot options
                        html.Div(
                            [
                                # Plot options
                                html.Div(
                                    [
                                        html.H5("Region"),
                                        dcc.Dropdown(
                                            id="pca_plot_region",
                                            options=[
                                                {"label": "CONUS", "value": "CONUS"},
                                            ],
                                            value="CONUS",
                                        ),
                                    ],
                                    className="two columns",
                                ),
                                html.Div(
                                    [
                                        html.H5("PCA Plot (Top) Color Value"),
                                        dcc.Dropdown(
                                            id="pca_plot_value_1",
                                            options=[
                                                {"label": "None", "value": "None"},
                                            ],
                                            value="None",
                                        ),
                                    ],
                                    className="two columns",
                                ),
                                # PCA Plot 2 options
                                html.Div(
                                    [
                                        html.H5("PCA Plot (Bottom) Color Value"),
                                        dcc.Dropdown(
                                            id="pca_plot_value_2",
                                            options=[
                                                {"label": "None", "value": "None"},
                                            ],
                                            value="None",
                                        ),
                                    ],
                                    className="two columns",
                                ),
                                # Plot options
                                html.Div(
                                    [
                                        html.H5("Axis 1"),
                                        dcc.Dropdown(
                                            id="pca_plot_axis1",
                                            options=[
                                                {"label": "None", "value": "None"},
                                            ],
                                            value="None",
                                        ),
                                    ],
                                    className="two columns",
                                ),
                                # Plot options
                                html.Div(
                                    [
                                        html.H5("Axis 2"),
                                        dcc.Dropdown(
                                            id="pca_plot_axis2",
                                            options=[
                                                {"label": "None", "value": "None"},
                                            ],
                                            value="None",
                                        ),
                                    ],
                                    className="two columns",
                                ),
                                # Plot options
                                html.Div(
                                    [
                                        html.H5("Axis 3"),
                                        dcc.Dropdown(
                                            id="pca_plot_axis3",
                                            options=[
                                                {"label": "None", "value": "None"},
                                            ],
                                            value="None",
                                        ),
                                    ],
                                    className="two columns",
                                ),
                                # Plot options
                                html.Div(
                                    [
                                        html.H5("Plot Value"),
                                        dcc.Dropdown(
                                            id="pca_plot_map_value",
                                            options=[
                                                {"label": "None", "value": "None"},
                                            ],
                                            value="None",
                                        ),
                                    ],
                                    className="two columns",
                                ),
                            ],
                            className="fourteen columns",
                        ),
                    ],
                    id="pca_scenarios",
                    className="eight columns",
                    style={"display": "none"},
                ),
            ]
        ),

        html.Hr(
            style={
                "width": "92%",
                "margin-left": "53px",
                "margin-right": "10px",
                "margin-bottom": "1px",
                "margin-top": "-1px",
                "border-top": "2px solid #fccd34",
                "border-bottom": "3px solid #1663b5",
            }
        ),

        # Submit Button to avoid repeated callbacks
        html.Div(
            [
                html.Button(
                    id="submit",
                    children="Submit",
                    style=BUTTON_STYLES["on"],
                    title=("Click to submit options"),
                    className="two columns",
                ),
            ],
            style={
                "margin-left": "50px",
                "margin-bottom": "25px",
                "margin-top": "2px",
            },
            className="row",
        ),

        # The chart and map div
        html.Div(
            [
                # The map div
                html.Div(
                    style={
                        "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)"
                    },
                    className="six columns",
                    children=[
                        # Above Map Options
                        above_map_options_div(id_prefix="map"),

                        # The map
                        map_div(id="map"),

                        # Below Map Options
                        below_map_options_div(
                            id_prefix="map",
                            className="eleven columns",
                        ),

                        # Loading State
                        html.Div(
                            [
                                dcc.Loading(id="map_loading"),
                            ],
                            className="twelve_columns",
                            style={"margin-top": "70px"},
                        ),
                    ],
                ),

                # The chart div
                html.Div(
                    style={
                        "box-shadow": " 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)"
                    },
                    className="six columns",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        # Chart options
                                        dcc.Tabs(
                                            id="chart_options_tab",
                                            value="chart",
                                            style=TAB_STYLE,
                                        ),

                                        # Type of chart
                                        html.Div(
                                            id="chart_options_div",
                                            children=[
                                                dcc.Dropdown(
                                                    id="chart_options",
                                                    clearable=False,
                                                    options=CHART_OPTIONS,
                                                    multi=False,
                                                    value="cumsum",
                                                )
                                            ],
                                        ),

                                        # X-axis Variable
                                        html.Div(
                                            id="chart_x_variable_options_div",
                                            children=[
                                                dcc.Dropdown(
                                                    id="chart_x_var_options",
                                                    clearable=False,
                                                    options=[
                                                        {
                                                            "label": "None",
                                                            "value": "None",
                                                        }
                                                    ],
                                                    multi=False,
                                                    value="capacity",
                                                )
                                            ],
                                        ),

                                        # Region grouping
                                        html.Div(
                                            id="chart_region_div",
                                            children=[
                                                dcc.Dropdown(
                                                    id="chart_region",
                                                    clearable=False,
                                                    options=REGION_OPTIONS,
                                                    multi=False,
                                                    value="national",
                                                )
                                            ],
                                        ),

                                        # Scenario grouping
                                        html.Div(
                                            id="additional_scenarios_div",
                                            children=[
                                                dcc.Dropdown(
                                                    id="additional_scenarios",
                                                    clearable=False,
                                                    options=[
                                                        {
                                                            "label": "None",
                                                            "value": "None",
                                                        }
                                                    ],
                                                    multi=True,
                                                )
                                            ],
                                        ),
                                    ]
                                ),
                            ],
                        ),

                        # The chart
                        html.Div(
                            children=dcc.Graph(
                                id="chart",
                                style={"height": 750},
                                config={
                                    "showSendToCloud": True,
                                    "toImageButtonOptions": {
                                        "width": 1250,
                                        "height": 750,
                                    },
                                    "plotlyServerURL": "https://chart-studio.plotly.com",
                                },
                                mathjax=True,
                                figure=go.Figure(
                                    layout={
                                        "xaxis": {"visible": False},
                                        "yaxis": {"visible": False},
                                        "annotations": [
                                            {
                                                "text": "No data loaded",
                                                "xref": "paper",
                                                "yref": "paper",
                                                "showarrow": False,
                                                "font": {"size": 28},
                                            }
                                        ],
                                    }
                                ),
                            ),
                        ),

                        # Below Chart Options
                        html.Div(
                            id="chart_extra_div",
                            children=[
                                html.P(
                                    "Point Size:",
                                    style={"display": "table-cell"}
                                ),
                                dcc.Input(
                                    id="chart_point_size",
                                    value=DEFAULT_POINT_SIZE,
                                    type="number",
                                    debounce=False,
                                    style={"width": "30%"}
                                ),
                                html.P(
                                    "Bin Size:",
                                    id="bin_size",
                                    style={"display": "none"}
                                ),
                                html.Div(
                                    id="chart_x_bin_div",
                                    style={"display": "none"},
                                    children=[
                                        dcc.Input(
                                            id="chart_x_bin",
                                            debounce=False,
                                            value=None,
                                            type="number",
                                            style={"width": "30%"}
                                        ),
                                    ],
                                ),
                                html.P(
                                    "Opacity:",
                                    style={"display": "table-cell"},
                                ),
                                html.Div(
                                    style={"display": "table-cell"},
                                    children=[
                                        dcc.Input(
                                            id="chart_alpha",
                                            value=1,
                                            type="number",
                                            debounce=False,
                                            style={"width": "30%"}                                        ),
                                    ]
                                ),
                            ],
                            className="twelve columns",
                            style=BOTTOM_DIV_STYLE,
                        ),

                        # Loading State
                        html.Div(
                            [
                                dcc.Loading(id="chart_loading"),
                            ],
                            className="row",
                            style={"margin-top": "70px"},
                        ),
                    ],
                ),
            ],
            className="row",
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
        html.Div(id="mapcap", style={"display": "none"}),
        # Filter list after being pieced together
        html.Div(id="filter_store", style={"display": "none"}),
    ]
)
