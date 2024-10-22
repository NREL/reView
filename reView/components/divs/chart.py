# -*- coding: utf-8 -*-
"""Chart Divs.

Functions for generating a chart divs with a given element type and class.
"""
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from dash import dcc, html

from reView.layout.options import CHART_OPTIONS
from reView.utils.constants import DEFAULT_POINT_SIZE
from reView.layout.styles import (
    OPTION_STYLE,
    OPTION_TITLE_STYLE,
    TAB_STYLE,
)


def above_chart_options_div(id_prefix, class_name="row"):
    """Build standard "above chart" options div.

    Parameters
    ----------
    id_prefix : str
        A string representing the prefix to use for individual
        components. Assuming `id_prefix="rev"`, the components are:
            - rev_map_options_tab
                A `dcc.Tabs` with tabs representing the map options for
                user.
            - rev_chart_state_options_div
                A `html.Div` that holds the `dcc.Dropdown` that includes
                all state options.
            - rev_chart_state_options
                A `dcc.Dropdown` to include all state options.
            - rev_chart_basemap_options_div
                A `html.Div` that holds the `dcc.Dropdown` that includes
                all basemap options.
            - rev_chart_basemap_options
                A `dcc.Dropdown` to include all basemap options.
            - rev_chart_color_options_div
                A `html.Div` that holds the `dcc.Dropdown` that includes
                all color options.
            - rev_chart_color_options
                A `dcc.Dropdown` to include all color options.
    class_name : str, optional
        The classname of the "above map" options div.
        By default, `None`.

    Returns
    -------
    dash.html.Div.Div
        A div containing the "below map" options that the user can
        interact with.
    """
    return html.Div(
        className=class_name,
        id="graph_options_div",
        children=[
            # Chart options
            dcc.Tabs(
                id=f"{id_prefix}_chart_options_tab",
                value="chart",
                style=TAB_STYLE,
            ),

            # Type of chart
            html.Div(
                id=f"{id_prefix}_chart_options_div",
                # className="seven columns",
                children=[
                    dcc.Dropdown(
                        id=f"{id_prefix}_chart_options",
                        clearable=False,
                        options=CHART_OPTIONS,
                        multi=False,
                        value="cumsum",
                    )
                ]
            ),

            # X-axis Variable
            html.Div(
                id=f"{id_prefix}_chart_x_variable_options_div",
                # className="six columns",
                children=[
                    dcc.Dropdown(
                        id=f"{id_prefix}_chart_x_var_options",
                        clearable=False,
                        options=[
                            {
                                "label": "None",
                                "value": "None",
                            }
                        ],
                        multi=False,
                        value="capacity_ac_mw",
                    )
                ]
            )
        ]
    )


def below_chart_options_div(id_prefix, class_name=None):
    """Build standard "below chart" options div.

    Parameters
    ----------
    id_prefix : str
        A string representing the prefix to use for individual
        components. Assuming `id_prefix="rev"`, the components are:
            - rev_map_point_size
                A `dcc.Input` for users to specify point size.
            - rev_map_color_min
                A `dcc.Input` for users to specify the minimum color
                scale value.
            - rev_map_color_max
                A `dcc.Input` for users to specify the maximum color
                scale value.
            - rev_map_rev_color
                An `html.Button` that users can click to request a
                colorscale reversal.
    class_name : str, optional
        The classname of the "below map" options div.
        By default, `None`.

    Returns
    -------
    dash.html.Div.Div
        A div containing the "below map" options that the user can
        interact with.
    """
    return dbc.Collapse(
        dbc.CardBody(
            children=[
                html.Div(
                    className="two columns",
                    children=[
                        html.P("POINT SIZE", style=OPTION_TITLE_STYLE),
                        dcc.Input(
                            id=f"{id_prefix}_chart_point_size",
                            value=DEFAULT_POINT_SIZE,
                            type="number",
                            debounce=False,
                            style=OPTION_STYLE,
                        ),
                    ],
                ),
                html.Div(
                    id=f"{id_prefix}_chart_x_bin_div",
                    className="two columns",
                    children=[
                        html.P("Bins", style=OPTION_TITLE_STYLE),
                        dcc.Input(
                            id=f"{id_prefix}_chart_x_bin",
                            debounce=False,
                            value=10,
                            type="number",
                            style=OPTION_STYLE,
                        ),
                    ],
                ),
                html.Div(
                    className="two columns",
                    children=[
                        html.P("Opacity:", style=OPTION_TITLE_STYLE),
                        dcc.Input(
                            id=f"{id_prefix}_chart_alpha",
                            value=1,
                            type="number",
                            debounce=False,
                            style=OPTION_STYLE,
                        ),
                    ],
                ),

                # Download Submission
                dbc.Button(
                    "DOWNLOAD",
                    id=f"{id_prefix}_chart_download_button",
                    className="me-1",
                    color="dark",
                    outline=True,
                    n_clicks=0,
                    style={
                        "float": "right",
                        "margin-right": "5px",
                        "margin-top": "-1px",
                        "color": "gray",
                        "border-color": "gray",
                    },
                )
            ]
        ),
        id=f"{id_prefix}_chart_below_options",
        className=class_name,
        is_open=False,
        style={"margin-top": "5px", "margin-left": "150px"},
    )


# pylint: disable=redefined-builtin,invalid-name
def chart_div(id_prefix, class_name=None):
    """Build standard reView chart div.

    Parameters
    ----------
    id : str
        A string representing the prefix of the Graph component that
        displays the map. The final id will be "<id_prefix>_map".
    class_name : str, optional
        The classname of the map div. By default, `None`.

    Returns
    -------
    dash.html.Div.Div
        A div containing the `dcc.Graph` component used to show the map.
    """
    return html.Div(
        children=[
            above_chart_options_div(id_prefix=id_prefix),
            dcc.Loading(
                id="rev_chart_loading",
                style={"margin-right": "500px"},
            ),
            dcc.Graph(
                id=f"{id_prefix}_chart",
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

            # Button to reveal below options
            dbc.Button(
                "Options",
                id=f"{id_prefix}_chart_below_options_button",
                className="mb-1",
                color="white",
                n_clicks=0,
                size="s",
                style={
                    "float": "left",
                    "margin-left": "15px",
                    "height": "50%",
                },
            ),
            below_chart_options_div(id_prefix, class_name="row"),
        ],
        className=class_name,
        style={
            "box-shadow": (
                "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 "
                "rgba(0, 0, 0, 0.19)"
            ),
            "border-radius": "5px",
        },
    )
