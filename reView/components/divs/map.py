# -*- coding: utf-8 -*-
"""A common map div."""
from dash import dcc, html
import plotly.graph_objects as go

from reView.utils.constants import DEFAULT_POINT_SIZE
from reView.layout.styles import BOTTOM_DIV_STYLE, RC_STYLES


# pylint: disable=redefined-builtin,invalid-name
def map_div(id, class_name=None):
    """Standard reView map div.

    Parameters
    ----------
    id : str
        ID to use for Graph component.
    class_name : str, optional
        The classname of the div. By default, `None`.

    Returns
    -------
    `html.Div`
        A div containing the `dcc.Graph` component used to show the map.
    """
    return html.Div(
        children=[
            dcc.Graph(
                id=id,
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
        ],
        className=class_name,
    )


def below_map_options_div(id_prefix, class_name=None):
    """Standard "below map" options div.

    Parameters
    ----------
    id_prefix : str
        A string representing the prefix to use for individual
        components. Assuming "prefix" is the inputs, the components are:
            - prefix_point_size
                A `dcc.Input` for users to specify point size.
            - prefix_color_min
                A `dcc.Input` for users to specify the minimum color
                scale value.
            - prefix_color_max
                A `dcc.Input` for users to specify the maximum color
                scale value.
            - prefix_rev_color
                An `html.Button` that users can click to request a
                colorscale reversal.
    class_name : str, optional
        The classname of the div. By default, `None`.

    Returns
    -------
    `html.Div`
        A div containing the "below map" options that the user can
        interact with.
    """

    return html.Div(
        [
            # Left options
            html.Div(
                [
                    html.P(
                        "Point Size:",
                        style={"margin-left": 5, "margin-top": 7},
                        className="two columns",
                    ),
                    dcc.Input(
                        id=f"{id_prefix}_point_size",
                        value=DEFAULT_POINT_SIZE,
                        type="number",
                        debounce=False,
                        className="one columns",
                        style={"margin-left": "-1px", "width": "10%"},
                    ),
                    html.P(
                        "Color Min: ",
                        style={"margin-top": 7},
                        className="two columns",
                    ),
                    dcc.Input(
                        id=f"{id_prefix}_color_min",
                        placeholder="",
                        type="number",
                        debounce=True,
                        className="one columns",
                        style={"margin-left": "-1px", "width": "10%"},
                    ),
                    html.P(
                        "Color Max: ",
                        style={"margin-top": 7},
                        className="two columns",
                    ),
                    dcc.Input(
                        id=f"{id_prefix}_color_max",
                        placeholder="",
                        debounce=True,
                        type="number",
                        className="one columns",
                        style={"margin-left": "-1px", "width": "10%"},
                    ),
                ],
                className="eight columns",
                style=BOTTOM_DIV_STYLE,
            ),

            # Right option
            html.Button(
                id=f"{id_prefix}_rev_color",
                children="Reverse Color: Off",
                n_clicks=0,
                type="button",
                title=(
                    "Click to render the map with the inverse "
                    "of the chosen color ramp."
                ),
                style=RC_STYLES["on"],
                className="one column",
            ),
        ],
        className=class_name,
    )
