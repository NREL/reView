# -*- coding: utf-8 -*-
"""ReEDS Buildout page callbacks.

Created on Mon May 23 21:07:15 2022

@author: twillia2
"""
import inspect
import json
import logging

import pandas as pd

from dash.dependencies import Input, Output

from reView.app import app

from reView.components.callbacks import toggle_reverse_color_button_style
from reView.components.map import Map
from reView.pages.reeds.model import cache_reeds
from reView.utils import calls
from reView.utils.functions import format_capacity_title

logger = logging.getLogger(__name__)
COMMON_CALLBACKS = [
    toggle_reverse_color_button_style(id_prefix="map_reeds"),
]


@app.callback(
    Output("capacity_print_reeds", "children"),
    Output("site_print_reeds", "children"),
    Input("mapcap_reeds", "children"),
    Input("map_reeds", "selectedData"),
)
@calls.log
def capacity_print(map_capacity, map_selection):
    """Calculate total remaining capacity after all filters are applied."""
    return format_capacity_title(map_capacity, map_selection)


@app.callback(
    Output("years_reeds", "value"),
    Output("years_reeds", "min"),
    Output("years_reeds", "max"),
    Output("years_reeds", "marks"),
    Input("project_reeds", "value"),
    Input("url", "pathname"),
)
@calls.log
def slider_year(project, url):
    """Return year slider for given project."""
    caller = inspect.stack()[0][3]
    logger.info("%s, args: %s", caller, f"{project=}, {url=}")

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
    Input("years_reeds", "value"),
    Input("map_reeds_rev_color", "n_clicks"),
)
@calls.log
def figure_map_reeds(project, year, reverse_color_clicks):
    """Return buildout table from single year as map."""

    caller = inspect.stack()[0][3]
    logger.info("%s, args: %s", caller, f"{project=}, {year=}")

    # Get data
    color_var = "capacity_MW"
    df = cache_reeds(project, year)
    df["print_capacity"] = df["capacity_MW"]

    agg = str(round(df[color_var].mean(), 2))
    title = f"Reference Advanced, 95% CO2 - {year} <br> Avg. {agg} MW"

    mapper = Map(df, color_var, title)
    figure = mapper.figure(
        point_size=4,
        reverse_color=reverse_color_clicks % 2 == 1,
    )

    mapcap = df[["sc_point_gid", "print_capacity"]].to_dict()

    return figure, json.dumps(mapcap)
