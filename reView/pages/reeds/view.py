# -*- coding: utf-8 -*-
"""The scenario page html layout.

Created on Tue Jul  6 15:23:09 2021

@author: twillia2
"""
from dash import dcc, html

import plotly.graph_objects as go

from reView.utils.functions import data_paths


PROJECT = str(list(data_paths()["reeds"].glob("*csv"))[0])


MAP_LAYOUT = dict(
    dragmode="select",
    font_family="Time New Roman",
    font_size=15,
    hovermode="closest",
    legend=dict(size=20),
    margin=dict(l=20, r=115, t=115, b=20),
    paper_bgcolor="#1663B5",
    plot_bgcolor="#083C04",
    titlefont=dict(color="white", size=18, family="Time New Roman"),
    title=dict(
        yref="container",
        x=0.05,
        y=0.95,
        yanchor="top",
        pad=dict(b=10),
    ),
    mapbox=dict(
        accesstoken=(
            "pk.eyJ1IjoidHJhdmlzc2l1cyIsImEiOiJjamZiaHh4b28waXNkMnpt"
            "aWlwcHZvdzdoIn0.9pxpgXxyyhM6qEF_dcyjIQ"
        ),
        style="satellite-streets",
        center=dict(lon=-97.5, lat=39.5),
        zoom=3.25,
    ),
)


layout = html.Div(
    children=[
        # Path Name
        dcc.Location(id="/reeds_page", refresh=False),

        # Constant info block
        html.Div(
            [
                # Project Selection
                html.Div(
                    [
                        html.H4("Project"),
                        dcc.Dropdown(
                            id="project_reeds",
                            options=[
                                # pylint: disable=not-an-iterable
                                {"label": "Reference Advanced - 95% CO",
                                 "value": PROJECT}
                            ],
                            value=PROJECT
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
                                html.H1(
                                    id="capacity_print_reeds",
                                    children=""
                                ),
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
                                html.H1(
                                    id="site_print_reeds",
                                    children=""
                                ),
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

        # Year selection
        html.P(
            "Year: ",
            id="year_text",
            className="four columns",
            style={"text-align": "left"}
        ),
        html.Div(
            [
                dcc.Slider(
                    id="years_reeds",
                    step=2
                )
            ],
            className="nine columns",
            style={"text-align": "center", "margin-bottom": "55px"}
        ),


        #                 # The map
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="map_reeds",
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
                            className="six columns"
                        ),

        # Capacity after make_map (avoiding duplicate calls)
        html.Div(id="mapcap_reeds", style={"display": "none"})
    ]
)
