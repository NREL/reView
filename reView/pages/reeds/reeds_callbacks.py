# -*- coding: utf-8 -*-
"""ReEDS Buildout page callbacks.

Created on Mon May 23 21:07:15 2022

@author: twillia2
"""
import hashlib
import inspect
import json
import logging
import os

from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from sklearn.metrics import DistanceMetric

from reView.app import app
from reView.layout.styles import BUTTON_STYLES, TABLET_STYLE, RC_STYLES
from reView.layout.options import (
    CHART_OPTIONS,
    COLOR_OPTIONS,
    COLOR_Q_OPTIONS,
)
from reView.pages.scenario.element_builders import (
    build_title,
    Map
)
from reView.pages.reeds.reeds_data import (
    cache_reeds,
    Map
)
from reView.utils.constants import SKIP_VARS
from reView.utils.functions import convert_to_title
from reView.utils.config import Config
from reView.utils import args

logger = logging.getLogger(__name__)


@app.callback(
    Output("capacity_print_reeds", "children"),
    Output("site_print_reeds", "children"),
    Input("mapcap_reeds", "children"),
    Input("map_reeds", "selectedData")
)
def capacity_print(mapcap, mapsel):
    """Calculate total remaining capacity after all filters are applied."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    # Calling this from make_map where the chartsel has already been applied
    nsites = ""
    capacity = ""
    if mapcap:
        df = pd.DataFrame(json.loads(mapcap))
        if not df.empty:
            if mapsel:
                gids = [p.get("customdata", [None])[0] for p in mapsel["points"]]
                df = df[df["sc_point_gid"].isin(gids)]
            nsites = "{:,}".format(df.shape[0])
            total_capacity = df["capacity_MW"].sum()
            if total_capacity >= 1_000_000:
                capacity = f"{round(total_capacity / 1_000_000, 4)} TW"
            else:
                capacity = f"{round(total_capacity / 1_000, 4)} GW"

    return capacity, nsites


@app.callback(
    Output("years_reeds", "value"),
    Output("years_reeds", "min"),
    Output("years_reeds", "max"),
    Output("years_reeds", "marks"),
    Input("project_reeds", "value"),
    Input("url", "pathname"),
)
def slider_year(project, url):
    """Return year slider for given project."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())
    caller = inspect.stack()[0][3]
    logger.info("%s, args: %s", caller, args.getargs(caller))

    # Get unique years from table
    years = pd.read_csv(project, usecols=["year"])["year"].unique()
    marks = {int(y): str(y) for y in years}
    ymin = int(years.min())
    ymax = int(years.max())
  
    return ymin, ymin, ymax, marks


@app.callback(
    Output("map_reeds", "figure"),
    Output("mapcap_reeds", "children"),
    Input("project_reeds", "value"),
    Input("years_reeds", "value")
)
def figure_map_reeds(project, year):
    """Return buildout table from single year as map."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())
    caller = inspect.stack()[0][3]
    logger.info("%s, args: %s", caller, args.getargs(caller))

    # Get data
    df = cache_reeds(project, year)
    mapper = Map(df, year)
    figure = mapper.figure

    return figure, json.dumps(mapper.mapcap)
