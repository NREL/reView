# -*- coding: utf-8 -*-
"""The scenario page html layout.

Created on Tue Jul  6 15:23:09 2021

@author: twillia2
"""
import json

from dash import dcc
from dash import html

import plotly.graph_objects as go
import reView

from reView.layout.styles import (
    BOTTOM_DIV_STYLE,
    BUTTON_STYLES,
    RC_STYLES,
    TAB_STYLE,
    TAB_BOTTOM_SELECTED_STYLE,
    TABLET_STYLE,
    TABLET_STYLE_CLOSED,
)
from reView.layout.options import (
    BASEMAP_OPTIONS,
    CHART_OPTIONS,
    COLOR_OPTIONS,
    REGION_OPTIONS,
    STATE_OPTIONS,
)
from reView.utils.constants import DEFAULT_POINT_SIZE
from reView.utils.classes import DiffUnitOptions
from reView.environment.settings import IS_DEV_ENV


PROJECT = str(list(reView.Paths.paths["reeds"].glob("*csv"))[0])  # <--------------- Fill in with list of options


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

        # # The chart and map div
        # html.Div(
        #     [
        #         # The map div
        #         html.Div(
        #             [
        #                 html.Div(
        #                     [
        #                         # Map options
        #                         dcc.Tabs(
        #                             id="map_options_tab_reeds",
        #                             value="state",
        #                             style=TAB_STYLE,
        #                             children=[
        #                                 dcc.Tab(
        #                                     value="state",
        #                                     label="State",
        #                                     style=TABLET_STYLE,
        #                                     selected_style=TABLET_STYLE,
        #                                 ),
        #                                 dcc.Tab(
        #                                     value="region",
        #                                     label="Region",
        #                                     style=TABLET_STYLE,
        #                                     selected_style=TABLET_STYLE,
        #                                 ),
        #                                 dcc.Tab(
        #                                     value="basemap",
        #                                     label="Basemap",
        #                                     style=TABLET_STYLE,
        #                                     selected_style=TABLET_STYLE,
        #                                 ),
        #                                 dcc.Tab(
        #                                     value="color",
        #                                     label="Color Ramp",
        #                                     style=TABLET_STYLE,
        #                                     selected_style=TABLET_STYLE,
        #                                 ),
        #                             ],
        #                         ),
        #                         # State options
        #                         html.Div(
        #                             id="state_options_div",
        #                             children=[
        #                                 dcc.Dropdown(
        #                                     id="state_options",
        #                                     clearable=True,
        #                                     options=STATE_OPTIONS,
        #                                     multi=True,
        #                                     value=None,
        #                                 )
        #                             ],
        #                         ),
        #                         html.Div(
        #                             id="region_options_div",
        #                             children=[
        #                                 dcc.Dropdown(
        #                                     id="region_options",
        #                                     clearable=True,
        #                                     options=REGION_OPTIONS,
        #                                     multi=True,
        #                                     value=None,
        #                                 )
        #                             ],
        #                         ),
        #                         # Basemap options
        #                         html.Div(
        #                             id="basemap_options_div",
        #                             children=[
        #                                 dcc.Dropdown(
        #                                     id="basemap_options",
        #                                     clearable=False,
        #                                     options=BASEMAP_OPTIONS,
        #                                     multi=False,
        #                                     value="light",
        #                                 )
        #                             ],
        #                         ),
        #                         # Color scale options
        #                         html.Div(
        #                             id="color_options_div",
        #                             children=[
        #                                 dcc.Dropdown(
        #                                     id="color_options",
        #                                     clearable=False,
        #                                     options=COLOR_OPTIONS,
        #                                     multi=False,
        #                                     value="Viridis",
        #                                 )
        #                             ],
        #                         ),
        #                     ],
        #                     className="row",
        #                 ),

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

        #                 # Below Map Options
        #                 html.Div(
        #                     [
        #                         # Left options
        #                         html.Div(
        #                             [
        #                                 html.P(
        #                                     "Point Size:",
        #                                     style={
        #                                         "margin-left": 5,
        #                                         "margin-top": 7,
        #                                     },
        #                                     className="two columns",
        #                                 ),
        #                                 dcc.Input(
        #                                     id="map_point_size",
        #                                     value=DEFAULT_POINT_SIZE,
        #                                     type="number",
        #                                     debounce=False,
        #                                     className="one columns",
        #                                     style={
        #                                         "margin-left": "-1px",
        #                                         "width": "10%",
        #                                     },
        #                                 ),
        #                                 html.P(
        #                                     "Color Min: ",
        #                                     style={"margin-top": 7},
        #                                     className="two columns",
        #                                 ),
        #                                 dcc.Input(
        #                                     id="map_color_min",
        #                                     placeholder="",
        #                                     type="number",
        #                                     debounce=True,
        #                                     className="one columns",
        #                                     style={
        #                                         "margin-left": "-1px",
        #                                         "width": "10%",
        #                                     },
        #                                 ),
        #                                 html.P(
        #                                     "Color Max: ",
        #                                     style={"margin-top": 7},
        #                                     className="two columns",
        #                                 ),
        #                                 dcc.Input(
        #                                     id="map_color_max",
        #                                     placeholder="",
        #                                     debounce=True,
        #                                     type="number",
        #                                     className="one columns",
        #                                     style={
        #                                         "margin-left": "-1px",
        #                                         "width": "10%",
        #                                     },
        #                                 ),
        #                             ],
        #                             className="eight columns",
        #                             style=BOTTOM_DIV_STYLE,
        #                         ),

        #                         # Right option
        #                         html.Button(
        #                             id="rev_color",
        #                             children="Reverse Color: Off",
        #                             n_clicks=0,
        #                             type="button",
        #                             title=(
        #                                 "Click to render the map with the inverse "
        #                                 "of the chosen color ramp."
        #                             ),
        #                             style=RC_STYLES["on"],
        #                             className="one column",
        #                         ),
        #                     ]
        #                 ),

        #                 # Loading State
        #                 html.Div(
        #                     [
        #                         dcc.Loading(
        #                             id="map_loading"
        #                         ),
        #                     ],
        #                     className="twelve_columns",
        #                     style={"margin-top": "70px"}
        #                 ),
        #             ],
        #             className="six columns",
        #         ),

        #         # The chart div
        #         html.Div(
        #             [
        #                 html.Div(
        #                     [
        #                         html.Div(
        #                             [
        #                                 # Chart options
        #                                 dcc.Tabs(
        #                                     id="chart_options_tab",
        #                                     value="chart",
        #                                     style=TAB_STYLE,
        #                                 ),
        #                                 # Type of chart
        #                                 html.Div(
        #                                     id="chart_options_div",
        #                                     children=[
        #                                         dcc.Dropdown(
        #                                             id="chart_options",
        #                                             clearable=False,
        #                                             options=CHART_OPTIONS,
        #                                             multi=False,
        #                                             value="cumsum",
        #                                         )
        #                                     ],
        #                                 ),
        #                                 # X-axis Variable
        #                                 html.Div(
        #                                     id="chart_xvariable_options_div",
        #                                     children=[
        #                                         dcc.Dropdown(
        #                                             id="chart_xvariable_options",
        #                                             clearable=False,
        #                                             options=[
        #                                                 {
        #                                                     "label": "None",
        #                                                     "value": "None",
        #                                                 }
        #                                             ],
        #                                             multi=False,
        #                                             value="capacity",
        #                                         )
        #                                     ],
        #                                 ),
        #                                 # Region grouping
        #                                 html.Div(
        #                                     id="chart_region_div",
        #                                     children=[
        #                                         dcc.Dropdown(
        #                                             id="chart_region",
        #                                             clearable=False,
        #                                             options=REGION_OPTIONS,
        #                                             multi=False,
        #                                             value="national",
        #                                         )
        #                                     ],
        #                                 ),
        #                                 # Scenario grouping
        #                                 html.Div(
        #                                     id="additional_scenarios_div",
        #                                     children=[
        #                                         dcc.Dropdown(
        #                                             id="additional_scenarios",
        #                                             clearable=False,
        #                                             options=[
        #                                                 {
        #                                                     "label": "None",
        #                                                     "value": "None",
        #                                                 }
        #                                             ],
        #                                             multi=True,
        #                                         )
        #                                     ],
        #                                 ),
        #                             ]
        #                         ),
        #                     ],
        #                     className="row",
        #                 ),
        #                 # The chart
        #                 html.Div(
        #                     children=dcc.Graph(
        #                         id="chart",
        #                         style={"height": 750},
        #                         config={
        #                             "showSendToCloud": True,
        #                             "toImageButtonOptions": {
        #                                 "width": 1250,
        #                                 "height": 750,
        #                             },
        #                             "plotlyServerURL": "https://chart-studio.plotly.com",
        #                         },
        #                         mathjax=True,
        #                         figure=go.Figure(
        #                             layout={
        #                                 "xaxis": {"visible": False},
        #                                 "yaxis": {"visible": False},
        #                                 "annotations": [
        #                                     {
        #                                         "text": "No data loaded",
        #                                         "xref": "paper",
        #                                         "yref": "paper",
        #                                         "showarrow": False,
        #                                         "font": {"size": 28},
        #                                     }
        #                                 ],
        #                             }
        #                         ),
        #                     ),
        #                 ),

        #                 # Below Chart Options
        #                 html.Div(
        #                     id="chart_extra_div",
        #                     children=[
        #                         html.P(
        #                             "Point Size:",
        #                             style={
        #                                 "margin-left": 5,
        #                                 "margin-top": 7,
        #                             },
        #                             className="three columns",
        #                         ),
        #                         dcc.Input(
        #                             id="chart_point_size",
        #                             value=DEFAULT_POINT_SIZE,
        #                             type="number",
        #                             debounce=False,
        #                             className="two columns",
        #                             style={"margin-left": "-1px"},
        #                         ),
        #                         html.Div(
        #                             id="chart_xbin_div",
        #                             style={"margin-left": "10px"},
        #                             children=[
        #                                 html.P(
        #                                     "Bin Size:",
        #                                     style={
        #                                         "margin-top": 7,
        #                                         "margin-left": 5,
        #                                     },
        #                                     className="three columns",
        #                                 ),
        #                                 dcc.Input(
        #                                     className="two columns",
        #                                     style={"margin-left": 5},
        #                                     id="chart_xbin",
        #                                     debounce=False,
        #                                     value=None,
        #                                     type="number",
        #                                 ),
        #                             ],
        #                         ),
        #                         html.Div(
        #                             [
        #                                 html.P(
        #                                     "Opacity:",
        #                                     style={
        #                                         "margin-left": 5,
        #                                         "margin-top": 7,
        #                                     },
        #                                     className="three columns",
        #                                 ),
        #                                 dcc.Input(
        #                                     id="chart_alpha",
        #                                     value=1,
        #                                     type="number",
        #                                     debounce=False,
        #                                     className="two columns",
        #                                     style={"margin-left": "-1px"},
        #                                 ),
        #                             ]
        #                         ),
        #                     ],
        #                     className="five columns",
        #                     style=BOTTOM_DIV_STYLE,
        #                 ),

        #                 # Loading State
        #                 html.Div(
        #                     [
        #                         dcc.Loading(
        #                             id="chart_loading"
        #                         ),
        #                     ],
        #                     className="row",
        #                     style={"margin-top": "70px"}
        #                 ),
        #             ],
        #             className="six columns",
        #         ),
        #     ],
        #     className="row",
        # ),

        # # To store option names for the map title
        # html.Div(id="chosen_map_options", style={"display": "none"}),

        # # To store option names for the chart title
        # html.Div(id="chosen_chart_options", style={"display": "none"}),

        # # For storing the data frame path and triggering updates
        # html.Div(id="map_data_path", style={"display": "none"}),

        # # For storing the signal need for the set of chart data frames
        # html.Div(id="chart_data_signal", style={"display": "none"}),

        # # Interim way to share data between map and chart
        # html.Div(id="map_signal", style={"display": "none"}),

        # # This table of recalc parameters
        # html.Div(
        #     id="recalc_table",
        #     children=json.dumps(
        #         {
        #             "scenario_a": {
        #                 "fcr": None,
        #                 "capex": None,
        #                 "opex": None,
        #                 "losses": None,
        #             },
        #             "scenario_b": {
        #                 "fcr": None,
        #                 "capex": None,
        #                 "opex": None,
        #                 "losses": None,
        #             },
        #         }
        #     ),
        #     style={"display": "none"},
        # ),

        # Capacity after make_map (avoiding duplicate calls)
        html.Div(id="mapcap_reeds", style={"display": "none"}),

        # # Filter list after being pieced together
        # html.Div(id="filter_store", style={"display": "none"}),
    ]
)
