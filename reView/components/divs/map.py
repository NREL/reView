# -*- coding: utf-8 -*-
"""A common map div."""
import dash_bootstrap_components as  dbc
import plotly.graph_objects as go

from dash import dcc, html

from reView.utils.constants import DEFAULT_POINT_SIZE
from reView.layout.styles import (
    BOTTOM_DIV_STYLE,
    OPTION_STYLE,
    OPTION_TITLE_STYLE,
    RC_STYLES,
    TAB_STYLE,
    TABLET_STYLE,
)
from reView.layout.options import (
    BASEMAP_OPTIONS,
    COLOR_OPTIONS,
    REGION_OPTIONS,
    STATE_OPTIONS,
)


def above_map_options_div(id_prefix, class_name=None):
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
    `html.Div`
        A div containing the "below map" options that the user can
        interact with.
    """

    return html.Div(
        [
            dcc.Tabs(
                id=f"{id_prefix}_map_options_tab",
                value="state",
                style=TAB_STYLE,
                children=[
                    dcc.Tab(
                        value="state",
                        label="State",
                        style=TABLET_STYLE,
                        selected_style=TABLET_STYLE,
                    ),
                    dcc.Tab(
                        value="region",
                        label="Region",
                        style=TABLET_STYLE,
                        selected_style=TABLET_STYLE,
                    ),
                    dcc.Tab(
                        value="basemap",
                        label="Basemap",
                        style=TABLET_STYLE,
                        selected_style=TABLET_STYLE,
                    ),
                    dcc.Tab(
                        value="color",
                        label="Color Ramp",
                        style=TABLET_STYLE,
                        selected_style=TABLET_STYLE,
                    ),
                ],
            ),
            # State options
            html.Div(
                id=f"{id_prefix}_map_state_options_div",
                children=[
                    dcc.Dropdown(
                        id=f"{id_prefix}_map_state_options",
                        clearable=True,
                        options=STATE_OPTIONS,
                        multi=True,
                        value=None,
                    )
                ],
            ),
            html.Div(
                id=f"{id_prefix}_map_region_options_div",
                children=[
                    dcc.Dropdown(
                        id=f"{id_prefix}_map_region_options",
                        clearable=True,
                        options=REGION_OPTIONS,
                        multi=True,
                        value=None,
                    )
                ],
            ),
            # Basemap options
            html.Div(
                id=f"{id_prefix}_map_basemap_options_div",
                children=[
                    dcc.Dropdown(
                        id=f"{id_prefix}_map_basemap_options",
                        clearable=False,
                        options=BASEMAP_OPTIONS,
                        multi=False,
                        value="light",
                    )
                ],
            ),
            # Color scale options
            html.Div(
                id=f"{id_prefix}_map_color_options_div",
                children=[
                    dcc.Dropdown(
                        id=f"{id_prefix}_map_color_options",
                        clearable=False,
                        options=COLOR_OPTIONS,
                        multi=False,
                        value="Viridis",
                    )
                ],
            ),
        ],
        className=class_name,
    )


# pylint: disable=redefined-builtin,invalid-name
def map_div(id_prefix, class_name=None):
    """Standard reView map div.

    Parameters
    ----------
    id : str
        A string representing the prefix of the Graph component that
        displays the map. The final id will be "<id_prefix>_map".
    class_name : str, optional
        The classname of the map div. By default, `None`.

    Returns
    -------
    `html.Div`
        A div containing the `dcc.Graph` component used to show the map.
    """
    return html.Div(
        children=[
            dcc.Graph(
                id=f"{id_prefix}_map",
                style={"height": 750},
                config={
                    "showSendToCloud": True,
                    "plotlyServerURL": "https://chart-studio.plotly.com",
                    "toImageButtonOptions": {
                        "width": 1250,
                        "height": 750,
                        "filename": "custom_review_map",
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
                id=f"{id_prefix}_map_below_options_button",
                className="mb-1",
                color="white",
                n_clicks=0,
                size="s",
                style={
                    "float": "left",
                    "margin-left": "15px",
                    "height": "50%"
                }
            ),

            # Below Map Options
            below_map_options_div(
                id_prefix=id_prefix,
                class_name="twelve columns"
            ),
        ],
        className=class_name,
    )


def below_map_options_div(id_prefix, class_name=None):
    """Standard "below map" options div.

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
    `html.Div`
        A div containing the "below map" options that the user can
        interact with.
    """
    return dbc.Collapse(
        dbc.CardBody(
            children=[
                html.Div(
                    className="two columns",
                    children=[
                        html.P(
                            "POINT SIZE",
                            style=OPTION_TITLE_STYLE
                        ),
                        dcc.Input(
                            id=f"{id_prefix}_map_point_size",
                            value=DEFAULT_POINT_SIZE,
                            type="number",
                            debounce=False,
                            style=OPTION_STYLE,
                        )
                    ]
                ),
                html.Div(
                    className="two columns",
                    children=[
                        html.P(
                            "COLOR MIN",
                            style=OPTION_TITLE_STYLE
                        ),
                        dcc.Input(
                            id=f"{id_prefix}_map_color_min",
                            placeholder="",
                            type="number",
                            debounce=True,
                            style=OPTION_STYLE,
                        ),
                    ]
                ),
               html.Div(
                   className="two columns",
                   children=[
                       html.P(
                           "COLOR MAX",
                           style=OPTION_TITLE_STYLE
                       ),
                       dcc.Input(
                           id=f"{id_prefix}_map_color_max",
                           placeholder="",
                           type="number",
                           debounce=True,
                           style=OPTION_STYLE,
                       ),
                   ]
               ),
               html.Div(
                   className="two columns",
                   children=[
                       html.P(
                           "REVERSE COLOR",
                           style=OPTION_TITLE_STYLE
                       ),
                       dbc.Button(
                           "    CLICK    ",
                           id=f"{id_prefix}_map_rev_color",
                           className="me-1",
                           color="dark",
                           outline=True,
                           n_clicks=0,
                           size="lg",
                           style={
                               "height": "98%",
                               "margin-top": "-1px",
                               "color": "gray",
                               "border-color": "gray",
                               "width": "125%"
                            }
                       ),
                   
                    ]
                ),

               # # Right option
               # html.Button(
               #     id=f"{id_prefix}_map_rev_color",
               #     children="Reverse Color: Off",
               #     n_clicks=0,
               #     type="button",
               #     title=(
               #         "Click to render the map with the inverse "
               #         "of the chosen color ramp."
               #     ),
               #     style=RC_STYLES["on"],
               #     className="one column",
               # ),
           ]
        ),
        id=f"{id_prefix}_map_below_options",
        className="row",
        is_open=False,
        style={"margin-top": "5px", "margin-left": "150px"}
    )
