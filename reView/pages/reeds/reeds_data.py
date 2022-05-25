# -*- coding: utf-8 -*-
"""ReEDS Buildout page data functions.

Created on Mon May 23 21:48:04 2022

@author: twillia2
"""
import json
import copy
import logging
import multiprocessing as mp
import operator
import os

from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.neighbors import BallTree
from tqdm import tqdm

from reView.pages.scenario.scenario import MAP_LAYOUT
from reView.utils.constants import AGGREGATIONS, DEFAULT_POINT_SIZE
from reView.utils.functions import (
    convert_to_title,
    strip_rev_filename_endings,
    lcoe,
    lcot,
    as_float,
    safe_convert_percentage_to_decimal,
    capacity_factor_from_lcoe,
    adjust_cf_for_losses,
    common_numeric_columns
)
from reView.utils.classes import DiffUnitOptions
from reView.utils.config import Config
from reView.utils.constants import COLORS
from reView.app import cache4

pd.set_option("mode.chained_assignment", None)
logger = logging.getLogger(__name__)


@cache4.memoize()
def cache_reeds(path, year):
    """Create table of single year buildout."""
    df = pd.read_csv(path)
    df = df[df["year"] == year]
    return df


class Map:
    """Initialize Map builder for ReEDS. Merge with scenario Map."""

    def __init__(
            self,
            df,
            year=2010,
            y="capacity_MW",
            basemap="light",
            chartsel=None,
            clicksel=None,
            color="Viridis",
            mapsel=None,
            point_size=4, 
            rev_color=False,
            uymin=None,
            uymax=None,
            title_size=18
        ):
        """Initialize ScatterPlot object."""
        self.df = df
        self.y = y
        self.year = year
        self.basemap = basemap
        self.chartsel = chartsel
        self.clicksel = clicksel
        self.color = color
        self.mapsel = mapsel
        self.point_size = point_size
        self.rev_color = rev_color
        self.title_size = title_size
        self.uymax = uymax 
        self.uymin = uymin

    def __repr__(self):
        """Return representation string."""
        name = self.__class__.__name__
        params = [f"{k}={v}" for k, v in self.__dict__.items() if k != "df"]
        params.append(f"df='dataframe with {self.df.shape[0]:,} rows'")
        param_str = ", ".join(params)
        msg = f"<{name} object: {param_str}>"
        return msg

    @property
    def figure(self):
        """Return scattermapbox figure."""
        self.df["hover"] = self.hover_text
        figure = px.scatter_mapbox(
            data_frame=self.df,
            lon="longitude",
            lat="latitude",
            hover_name="hover"
        )
        figure.update_traces(marker=self.marker)
        figure.update_layout(**self.layout)
        return figure

    @property
    def hover_text(self):
        """Return hover text column."""
        df = self.df
        y = self.y
        return round(df[y], 2).astype(str)

    @property
    def layout(self):
        """Build the map data layout dictionary."""
        layout = copy.deepcopy(MAP_LAYOUT)
        layout["mapbox"]["style"] = self.basemap
        layout["showlegend"] = False
        layout["title"]["text"] = self.title
        layout["uirevision"] = True
        layout["yaxis"] = dict(range=[0, 400])
        layout["legend"] = dict(
            title_font_family="Times New Roman",
            bgcolor="#E4ECF6",
            font=dict(family="Times New Roman", size=15, color="black"),
        )
        return layout

    @property
    def marker(self):
        """Return marker dictionary."""
        pcolor = COLORS[self.color]
        marker = dict(
            color=self.df[self.y],
            colorscale=pcolor,
            opacity=1.0,
            reversescale=self.rev_color,
            size=self.point_size
        )

        return marker

    @property
    def title(self):
        """Return figure title."""
        agg = self.df[self.y].mean()
        agg = str(round(agg, 2))
        return f"Reference Advanced, 95% CO2 - {self.year} <br> Avg. {agg} MW"

    @property
    def mapcap(self):
        """Return total capacity."""
        return self.df[["sc_point_gid", "capacity_MW"]].to_dict()
