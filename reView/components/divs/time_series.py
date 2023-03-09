# -*- coding: utf-8 -*-
"""A common time series div."""
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from dash import dcc, html

from reView.utils.constants import DEFAULT_POINT_SIZE
from reView.layout.styles import (
    OPTION_STYLE,
    OPTION_TITLE_STYLE,
    TAB_STYLE,
    TABLET_STYLE,
)
from reView.layout.options import (
    BASEMAP_OPTIONS,
    COLOR_OPTIONS,
    REGION_OPTIONS,
    STATE_OPTIONS,
)


def above_time_options_div(id_prefix, class_name=None):
    """Standard "above map" options div.

    Parameters
    ----------
    id_prefix : str
        A string representing the prefix to use for individual
        components. Assuming `id_prefix="rev"`, the components are:
            - rev_map_options_tab
                A `dcc.Tabs` with tabs representing the map options for
                user.
            - rev_map_state_options_div
                A `html.Div` that holds the `dcc.Dropdown` that includes
                all state options.
            - rev_map_state_options
                A `dcc.Dropdown` to include all state options.
            - rev_map_region_options_div
                A `html.Div` that holds the `dcc.Dropdown` that includes
                all region options.
            - rev_map_region_options
                A `dcc.Dropdown` to include all region options.
            - rev_map_basemap_options_div
                A `html.Div` that holds the `dcc.Dropdown` that includes
                all basemap options.
            - rev_map_basemap_options
                A `dcc.Dropdown` to include all basemap options.
            - rev_map_color_options_div
                A `html.Div` that holds the `dcc.Dropdown` that includes
                all color options.
            - rev_map_color_options
                A `dcc.Dropdown` to include all color options.
    class_name : str, optional
        The classname of the "above map" options div.
        By default, `None`.

    Returns
    -------
    dash.html.Div
        A div containing the "below map" options that the user can
        interact with.
    """
    return html.Div(
        className=class_name,
        children=[
            # Trace Type Options
            dcc.Tabs(
                id=f"{id_prefix}_time_trace_options_tab",
                value="bar",
                style=TAB_STYLE,
                children=[
                    dcc.Tab(
                        value="line",
                        label="Line",
                        style=TABLET_STYLE,
                        selected_style=TABLET_STYLE,
                    ),
                    dcc.Tab(
                        value="bar",
                        label="Bar",
                        style=TABLET_STYLE,
                        selected_style=TABLET_STYLE,
                    )
                ],
            ),

            # Time window options
            dcc.Tabs(
                id=f"{id_prefix}_time_period_options_tab",
                value="original",
                style=TAB_STYLE,
                children=[
                    dcc.Tab(
                        value="original",
                        label="Original",
                        style=TABLET_STYLE,
                        selected_style=TABLET_STYLE,
                    ),
                    dcc.Tab(
                        value="hour",
                        label="Diurnal",
                        style=TABLET_STYLE,
                        selected_style=TABLET_STYLE,
                    ),
                    dcc.Tab(
                        value="daily",
                        label="Daily",
                        style=TABLET_STYLE,
                        selected_style=TABLET_STYLE,
                    ),
                    dcc.Tab(
                        value="weekly",
                        label="Weekly",
                        style=TABLET_STYLE,
                        selected_style=TABLET_STYLE,
                    ),
                    dcc.Tab(
                        value="monthly",
                        label="Monthly",
                        style=TABLET_STYLE,
                        selected_style=TABLET_STYLE,
                    )
                ],
            ),

            # Placeholder First Options
            html.Div(
                children=[
                    dcc.Dropdown(
                        id=f"{id_prefix}_additional_scenarios_time",
                        clearable=False,
                        placeholder="Additional Scenarios",
                        multi=True
                    )
                ]
            ),

            # Placeholder Second Options
            html.Div(
                id=f"{id_prefix}_options_2",
                style={"display": "none"},
                children=[
                    dcc.Dropdown(
                        id=f"{id_prefix}_option_2_options",
                        clearable=False,
                        # options=COLOR_OPTIONS,
                        multi=False,
                        value="Viridis",
                    )
                ],
            ),
        ],
    )


# pylint: disable=redefined-builtin,invalid-name
def time_div(id_prefix, class_name=None):
    """Standard reView time div.

    Parameters
    ----------
    id : str
        A string representing the prefix of the Graph component that
        displays the timeseries. The final id will be "<id_prefix>_time".
    class_name : str, optional
        The classname of the time div. By default, `None`.

    Returns
    -------
    dash.html.Div
        A div containing the `dcc.Graph` component used to show the timeseries.
    """
    return html.Div(
        children=[
            # Above Timeseries Options
            above_time_options_div(id_prefix=id_prefix),
            dcc.Loading(
                id=f"{id_prefix}_time_loading",
                style={"margin-right": "500px"},
            ),
            dcc.Graph(
                id=f"{id_prefix}_time",
                style={"height": 750},
                config={
                    "showSendToCloud": True,
                    "plotlyServerURL": "https://chart-studio.plotly.com",
                    "toImageButtonOptions": {
                        "width": 1250,
                        "height": 750,
                        "filename": "custom_review_time",
                    },
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
                id=f"{id_prefix}_time_below_options_button",
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

            # Below Timeseries Options
            below_time_options_div(id_prefix=id_prefix, class_name="row"),
        ],
        className=class_name,
        style={
            "margin-top": "50px",
            "box-shadow": (
                "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 "
                "rgba(0, 0, 0, 0.19)"
            ),
            "border-radius": "5px",
        },
    )


def below_time_options_div(id_prefix, class_name=None):
    """Standard "below time" options div.

    Parameters
    ----------
    id_prefix : str
        A string representing the prefix to use for individual
        components. Assuming `id_prefix="rev"`, the components are:
            - rev_time_point_size
                A `dcc.Input` for users to specify point size.
            - rev_time_color_min
                A `dcc.Input` for users to specify the minimum color
                scale value.
            - rev_time_color_max
                A `dcc.Input` for users to specify the maximum color
                scale value.
            - rev_time_rev_color
                An `html.Button` that users can click to request a
                colorscale reversal.
    class_name : str, optional
        The classname of the "below time" options div.
        By default, `None`.

    Returns
    -------
    dash.html.Div.Div
        A div containing the "below time" options that the user can
        interact with.
    """
    return dbc.Collapse(
        dbc.CardBody(
            children=[
                html.Div(
                    style={"justifyContent": "center"},
                    className="two columns",
                    children=[
                        html.P("POINT SIZE", style=OPTION_TITLE_STYLE),
                        dcc.Input(
                            id=f"{id_prefix}_time_point_size",
                            value=DEFAULT_POINT_SIZE,
                            type="number",
                            debounce=True,
                            style=OPTION_STYLE,
                        ),
                    ],
                ),
                html.Div(
                    className="two columns",
                    children=[
                        html.P("COLOR MIN", style=OPTION_TITLE_STYLE),
                        dcc.Input(
                            id=f"{id_prefix}_time_color_min",
                            placeholder="",
                            type="number",
                            debounce=True,
                            style=OPTION_STYLE,
                        ),
                    ],
                ),
                html.Div(
                    className="two columns",
                    children=[
                        html.P("COLOR MAX", style=OPTION_TITLE_STYLE),
                        dcc.Input(
                            id=f"{id_prefix}_time_color_max",
                            placeholder="",
                            type="number",
                            debounce=True,
                            style=OPTION_STYLE,
                        ),
                    ],
                ),
                dbc.Button(
                    "REVERSE COLOR",
                    id=f"{id_prefix}_time_rev_color",
                    className="me-1",
                    color="dark",
                    outline=True,
                    n_clicks=0,
                    # size="sm",
                    style={
                        "float": "right",
                        "margin-top": "-1px",
                        "color": "gray",
                        "border-color": "gray",
                    },
                ),

                # Download Submission
                dbc.Button(
                    "DOWNLOAD",
                    id=f"{id_prefix}_time_download_button",
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
        id=f"{id_prefix}_time_below_options",
        className=class_name,
        is_open=False,
        style={"margin-top": "5px", "margin-left": "150px"},
    )
