# -*- coding: utf-8 -*-
"""A common map div."""
from dash import dcc, html


REV_PCA_DIV = html.Div(
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
)