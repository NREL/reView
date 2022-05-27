# -*- coding: utf-8 -*-
"""A common map div."""
from dash import dcc, html
import plotly.graph_objects as go


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
