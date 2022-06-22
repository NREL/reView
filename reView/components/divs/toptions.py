# -*- coding: utf-8 -*-
"""A common map div."""
import dash_bootstrap_components as dbc

from dash import dcc, html

from reView.environment.settings import IS_DEV_ENV
from reView.layout.styles import (
    TAB_BOTTOM_SELECTED_STYLE,
    TAB_STYLE,
    TABLET_STYLE,
    TABLET_STYLE_CLOSED,
)
from reView.utils.config import Config


TOPTIONS = [
    dcc.Tabs(
        id="scenario_selection_tabs",
        value="0",
        children=[
            dcc.Tab(
                label="Default Scenario",
                value="0",
                style=TABLET_STYLE,
                selected_style=TABLET_STYLE,
            ),
            dcc.Tab(
                label="Lowest Scenario",
                value="1",
                style=TABLET_STYLE,
                selected_style=TABLET_STYLE,
            ),
            # dcc.Tab(
            #     label="PCA" if IS_DEV_ENV else "Under construction",
            #     value="2",
            #     disabled=not IS_DEV_ENV,
            #     style=TABLET_STYLE,
            #     selected_style=TABLET_STYLE,
            # ),
        ],
        style={"display": "none"},
    ),

    # Data Options
    html.Div(
        [
            html.Div(
                # className="six columns",
                style={"margin-bottom": "10px"},
                children=[
                    # Project Selection
                    html.Div(
                        [
                            html.H5("Project"),
                            dcc.Dropdown(
                                id="project",
                                options=[
                                    # pylint: disable=not-an-iterable
                                    {"label": project, "value": project}
                                    for project in Config.sorted_projects
                                ],
                            ),
                        ],
                        className="four columns",
                        style={"margin-left": "50px"}
                    ),

                    # First Scenario
                    html.Div(
                        [
                            html.H5("Scenario A"),
                            html.Div(
                                id="scenario_a_options",
                                children=[
                                    dcc.Dropdown(
                                        id="scenario_dropdown_a",
                                        value="placeholder"
                                    )
                                ]
                            ),
                            html.Div(
                                id="scenario_a_specs",
                                style={
                                    "overflow-y": "auto",
                                    "height": "300px",
                                    "width": "95%"
                                }
                            ),
                            html.Hr(style={"height": "1.5px", "width": "95%"})
                        ],
                        className="four columns",
                        style={"margin-left": "25px"},
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
                                        children=[
                                            dcc.Dropdown(
                                                id="scenario_dropdown_b",
                                                value="placeholder"
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        id="scenario_b_specs",
                                        style={
                                            "overflow-y": "auto",
                                            "height": "300px",
                                            "width": "95%"
                                        }
                                    ),
                                    html.Hr(
                                        style={
                                            "height": "1.5px",
                                            "width": "95%"}
                                    )
                                ],
                                className="four columns",
                                style={"margin-left": "5px"},
                            )
                        ],
                    ),
                ],
            ),

            # Variable options
            html.Div(
                [
                    html.H5("Variable"),
                    dcc.Dropdown(
                        id="variable",
                        options=[
                            {
                                "label": "Capacity",
                                "value": "capacity",
                            }
                        ],
                        value="capacity",
                    ),
                ],
                className="three columns",
            ),

            # Show difference map
            html.Div(
                [
                    html.H5(
                        "Scenario B Difference",
                        title="(SCENARIO_B - SCENARIO_A) / SCENARIO_A",
                    ),
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
                        value="original",
                        style=TAB_STYLE,
                        children=[
                            dcc.Tab(
                                value="percent",
                                label="Percentage",
                                style=TABLET_STYLE,
                                selected_style=TAB_BOTTOM_SELECTED_STYLE,
                            ),
                            dcc.Tab(
                                value="original",
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
                className="four columns",
            ),

            # Add in a map function option (demand meeting)
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
                id="recalc_table_div",
                className="four columns",
            ),

            # Filters
            html.Div(
                children=[
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
                    ),
                ],
                className="four columns",
            ),
        ],
        id="options",
        className="row",
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
                className="four columns",
                style={"margin-left": "25px"},
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
                className="three columns",
            ),

            # Plot options
            html.Div(
                [
                    html.H5("Plot Value"),
                    dcc.Dropdown(
                        id="minimizing_plot_value",
                        options=[
                            {
                                "label": "Variable",
                                "value": "Variable",
                            },
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
]


REV_TOPTIONS_DIV = html.Div(
    className="twelve columns",
    style={
        "box-shadow": (
            "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 "
            "rgba(0, 0, 0, 0.19)"
        ),
        "border-radius": "5px",
        "margin-bottom": "50px",
        "margin-top": "25px",
        "margin-left": "0px",
    },
    children=[
        # Scenario selection tabs - Tabs for selection options
        dbc.Collapse(
            className="twelve columns",
            id="options_div",
            is_open=False,
            children=TOPTIONS,
        ),
        html.Div(
            children=[
                html.H5(
                    id="options_label",
                    children="OPTIONS",
                    style={"display": "none"},
                ),
                html.Div(
                    className="eleven columns",
                    children=[
                        html.Hr(
                            style={
                                "color": "#1663B5",
                                "width": "100%",
                                "margin-left": "-50px",
                                "height": "3px",
                                "margin-bottom": "0px",
                                "opacity": "1",
                            },
                        ),
                        html.Hr(
                            style={
                                "color": "#FCCD34",
                                "width": "99%",
                                "margin-left": "-45px",
                                "height": "2px",
                                "margin-top": "0px",
                                "margin-bottom": "1px",
                                "opacity": "1",
                            },
                        ),
                    ],
                ),

                # Hide/show options
                dbc.Button(
                    id="toggle_options",
                    children="Hide",
                    color="white",
                    n_clicks=1,
                    size="lg",
                    title=("Click to display options"),
                    className="mb-1",
                    style={
                        "float": "left",
                        "margin-left": "15px",
                        "height": "50%",
                    },
                ),

                # Submit Button to avoid repeated callbacks
                dbc.Button(
                    id="submit",
                    children="Submit",
                    color="white",
                    n_clicks=0,
                    size="lg",
                    title=("Click to submit options"),
                    className="mb-1",
                    style={
                        "float": "left",
                        "margin-left": "15px",
                        "height": "50%",
                    },
                ),
            ]
        ),
    ],
)
