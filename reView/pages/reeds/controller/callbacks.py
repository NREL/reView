# -*- coding: utf-8 -*-
"""ReEDS Buildout page callbacks.

Created on Mon May 23 21:07:15 2022

@author: twillia2
"""
import json
import logging
import os

import pandas as pd

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from reView import paths
from reView.app import app
from reView.components.callbacks import (
    capacity_print,
    display_selected_tab_above_map,
)
from reView.components.map import Map
from reView.pages.reeds.model import cache_reeds
from reView.utils import calls

logger = logging.getLogger(__name__)
COMMON_CALLBACKS = [
    capacity_print(id_prefix="reeds_1"),
    capacity_print(id_prefix="reeds_2"),
    display_selected_tab_above_map(id_prefix="reeds_1"),
    display_selected_tab_above_map(id_prefix="reeds_2"),
]
CAPACITY_COLUMNS = ["capacity_MW", "built_capacity", "capacity"]


def to_name(path):
    """Convert a file path to a more readable name."""
    fname = os.path.splitext(os.path.basename(path))[0].upper()
    name = " ".join(fname.split("_"))
    return name


@app.callback(
    Output("years_reeds", "value"),
    Output("years_reeds", "min"),
    Output("years_reeds", "max"),
    Output("years_reeds", "marks"),
    Input("project_reeds_1", "value"),
    Input("url", "pathname"),
)
@calls.log
def slider_year(project, __):
    """Return year slider for given project."""
    # Get unique years from table
    years = pd.read_csv(project, usecols=["year"])["year"].unique()
    marks = {}
    for year in years:
        marks[int(year)] = {
            "label": str(year),
            "style": {"transform": "rotate(45deg)"}
        }

    ymin = int(years.min())
    ymax = int(years.max())

    return ymin, ymin, ymax, marks


@app.callback(
    Output("year_text", "children"),
    Input("years_reeds", "value")
)
def year_print(year):
    """Return year to year print out header."""
    return str(year)


# pylint: disable=cell-var-from-loop
for i in [1, 2]:

    @app.callback(
        Output(f"project_reeds_{i}", "options"),
        Output(f"project_reeds_{i}", "value"),
        Input("url", "pathname"),
        State(f"call_{i}", "children")
    )
    @calls.log
    def dropdown_projects_reeds(__, call):
        """Update reeds project options."""
        files = list(paths.paths["reeds"].glob("*csv"))
        files.sort()
        options = [
            {"label": to_name(file), "value": str(file)} for file in files
        ]
        call_number = int(call)
        return options, options[call_number]["value"]

    # pylint: disable=too-many-locals
    @app.callback(
        Output(f"reeds_{i}_map", "figure"),
        Output(f"reeds_{i}_mapcap", "children"),
        Input(f"project_reeds_{i}", "value"),
        Input("years_reeds", "value"),
        Input(f"reeds_{i}_map_basemap_options", "value"),
        Input(f"reeds_{i}_map_color_options", "value"),
        Input(f"reeds_{i}_map_point_size", "value"),
        Input(f"reeds_{i}_map_rev_color", "n_clicks"),
        Input(f"reeds_{i}_map_color_min", "value"),
        Input(f"reeds_{i}_map_color_max", "value"),
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
        # If there is no year, the dropdowns haven't populated yet
        if not year:
            raise PreventUpdate

        # Get data
        df = cache_reeds(project, year)
        capacity = [col for col in CAPACITY_COLUMNS if col in df.columns]
        if len(capacity) != 1:
            raise AssertionError("Could not find capacity column in "
                                 f"{project}.")
        capacity = capacity[0]
        df["print_capacity"] = df[capacity]

        # Build Title
        agg = str(round(df[capacity].mean(), 2))
        name = to_name(project)
        title = f"{name} - {year} <br> Avg. {agg} MW"

        mapper = Map(
            df=df,
            color_var=capacity,
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

    @app.callback(
        Output(f"reeds_{i}_map_below_options", "is_open"),
        Input(f"reeds_{i}_map_below_options_button", "n_clicks"),
        State(f"reeds_{i}_map_below_options", "is_open"),
    )
    @calls.log
    def toggle_reeds_map_below_options(n_clicks, is_open):
        """Toggle the blow options for reeds."""
        logger.debug("REEDS OPTIONS BUTTON TRIGGERED...")
        if n_clicks:
            return not is_open
        return is_open
