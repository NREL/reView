# -*- coding: utf-8 -*-
"""ReEDS Buildout page callbacks.

Created on Mon May 23 21:07:15 2022

@author: twillia2
"""
import json
import logging

import pandas as pd

from dash.dependencies import Input, Output

from reView.app import app

from reView.components.callbacks import (
    capacity_print,
    toggle_reverse_color_button_style,
    display_selected_tab_above_map,
)
from reView.components.map import Map
from reView.pages.reeds.model import cache_reeds
from reView.utils import calls

logger = logging.getLogger(__name__)
COMMON_CALLBACKS = [
    capacity_print(id_prefix="reeds"),
    toggle_reverse_color_button_style(id_prefix="reeds"),
    display_selected_tab_above_map(id_prefix="reeds"),
]


@app.callback(
    Output("years_reeds", "value"),
    Output("years_reeds", "min"),
    Output("years_reeds", "max"),
    Output("years_reeds", "marks"),
    Input("project_reeds", "value"),
    Input("url", "pathname"),
)
@calls.log
def slider_year(project, __):
    """Return year slider for given project."""

    # Get unique years from table
    years = pd.read_csv(project, usecols=["year"])["year"].unique()
    marks = {int(y): str(y) for y in years}
    ymin = int(years.min())
    ymax = int(years.max())

    return ymin, ymin, ymax, marks


@app.callback(
    Output("reeds_map", "figure"),
    Output("reeds_mapcap", "children"),
    Input("project_reeds", "value"),
    Input("years_reeds", "value"),
    Input("reeds_map_basemap_options", "value"),
    Input("reeds_map_color_options", "value"),
    Input("reeds_map_point_size", "value"),
    Input("reeds_map_rev_color", "n_clicks"),
    Input("reeds_map_color_min", "value"),
    Input("reeds_map_color_max", "value"),
)
@calls.log
def figure_map_reeds(
    project,
    year,
    basemap,
    color,
    point_size,
    reverse_color_clicks,
    color_ymin,
    color_ymax,
):
    """Return buildout table from single year as map."""

    # Get data
    color_var = "capacity_MW"
    df = cache_reeds(project, year)
    df["print_capacity"] = df["capacity_MW"]

    agg = str(round(df[color_var].mean(), 2))
    title = f"Reference Advanced, 95% CO2 - {year} <br> Avg. {agg} MW"

    mapper = Map(
        df=df,
        color_var=color_var,
        plot_title=title,
        basemap=basemap,
        colorscale=color,
        color_min=color_ymin,
        color_max=color_ymax,
    )
    figure = mapper.figure(
        point_size=point_size,
        reverse_color=reverse_color_clicks % 2 == 1,
    )

    mapcap = df[["sc_point_gid", "print_capacity"]].to_dict()

    return figure, json.dumps(mapcap)
