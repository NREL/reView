# -*- coding: utf-8 -*-
"""Map class.

Used in (at least) the scenario and reeds pages.
"""
import copy

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from reView import UNITS, Q_
from reView.utils.classes import DiffUnitOptions
from reView.utils.config import Config
from reView.utils.constants import COLORS, DEFAULT_LAYOUT
from reView.utils.functions import convert_to_title

MAP_LAYOUT = copy.deepcopy(DEFAULT_LAYOUT)
MAP_LAYOUT.update(
    {
        "margin": {"l": 20, "r": 115, "t": 70, "b": 20},
        "plot_bgcolor": "#083C04",
        "mapbox": {
            "accesstoken": (
                "pk.eyJ1IjoidHJhdmlzc2l1cyIsImEiOiJjamZiaHh4b28waXNkMnpt"
                "aWlwcHZvdzdoIn0.9pxpgXxyyhM6qEF_dcyjIQ"
            ),
            "style": "satellite-streets",
            "center": {"lon": -97.5, "lat": 39.5},
            "zoom": 2.75,
        },
        "uirevision": True,
    }
)


def build_title(df, var, project, map_selection=None, delimiter="  |  "):
    """Create chart title."""
    # Project configuration object
    config = Config(project)
    var_no_diff_suffix = DiffUnitOptions.remove_from_variable_name(var)
    diff = DiffUnitOptions.from_variable_name(var) is not None
    is_percentage_diff = (
        DiffUnitOptions.from_variable_name(var) == DiffUnitOptions.PERCENTAGE
    )
    if diff and is_percentage_diff:
        units = "percent"
    else:
        units = config.units.get(var_no_diff_suffix)

    # Append variable title
    title = config.titles.get(var_no_diff_suffix, convert_to_title(var))

    # Difference title
    if diff:
        title = delimiter.join([title, "Difference"])

    is_df = isinstance(df, pd.core.frame.DataFrame)
    var_exists = var_no_diff_suffix and var_no_diff_suffix.lower() != "none"
    not_category = units != "category"

    # Map title (not chart)
    if is_df and var_exists and not_category:
        average = Q_(df[var].apply("mean"), units)
        if average.dimensionless:
            average = average.to_reduced_units()
        if "dollar" not in f"{average}":
            average = average.to_compact()

        extra = f"Average: {average:~H.2f}"

        # we can make this more general by
        # allowing user input about this in config
        if "capacity" in var_no_diff_suffix and units != "percent":
            capacity = (df[var].apply("sum") * UNITS.MW).to_compact()
            extra = delimiter.join([extra, f"Total: {capacity:~H.2f}"])

        if "hydrogen_annual_kg" in df and not diff:
            total_hydrogen = df["hydrogen_annual_kg"].sum() * UNITS.kilograms
            extra = delimiter.join(
                [extra, f"Total H2: {total_hydrogen.to_compact():~H.2f}"]
            )

        title = delimiter.join([title, extra])

    if map_selection:
        n_points_selected = len(map_selection["points"])
        map_selection_print = f"Selected point count: {n_points_selected:,}"
        title = delimiter.join([title, map_selection_print])

    return title


class Map:
    """Methods for building the mapbox scatter plot."""

    def __init__(
        self,
        df,
        color_var,
        plot_title,
        project=None,
        basemap="light",
        colorscale="Viridis",
        color_min=None,
        color_max=None,
        demand_data=None,
    ):
        """Initialize ScatterPlot object."""
        self.df = df
        self.color_var = color_var
        self.plot_title = plot_title
        self.project = project
        self.basemap = basemap
        self.colorscale = colorscale
        self.cmin, self.cmax = ColorRange(
            df, color_var, project, color_min, color_max
        )
        self.demand_data = demand_data

        if project:
            self.units = Config(self.project).units.get(self.color_var, "")
        else:
            self.units = ""

    def __repr__(self):
        """Return representation string."""
        name = self.__class__.__name__
        params = [f"{k}={v}" for k, v in self.__dict__.items() if k != "df"]
        params.append(f"df='DataFrame with {self.df.shape[0]:,} rows'")
        param_str = ", ".join(params)
        msg = f"<{name} object: {param_str}>"
        return msg

    def figure(self, point_size, reverse_color=False):
        """Build scatter plot figure."""
        self.df["text"] = self.hover_text
        if self.df.empty:
            figure = go.Figure()
            figure.update_layout(
                xaxis={"visible": False},
                yaxis={"visible": False},
                annotations=[
                    {
                        "text": "No matching data found",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 28},
                    }
                ],
            )
        elif self.units == "category":
            # Create data object
            figure = px.scatter_mapbox(
                data_frame=self.df,
                color=self.color_var,
                lon="longitude",
                lat="latitude",
                custom_data=["sc_point_gid", "print_capacity"],
                hover_name="text",
            )
            figure.update_traces(marker=self.marker(point_size, reverse_color))
        else:
            # Create data object
            figure = px.scatter_mapbox(
                data_frame=self.df,
                lon="longitude",
                lat="latitude",
                custom_data=["sc_point_gid", "print_capacity"],
                hover_name="text",
            )
            figure.update_traces(marker=self.marker(point_size, reverse_color))

            if self.demand_data is not None:
                self.demand_data["text"] = (
                    self.demand_data["sera_node"]
                    + ", "
                    + self.demand_data["State"]
                    + ". <br>Demand:   "
                    + self.demand_data["load"].astype(str)
                    + " kg"
                )

                fig2 = px.scatter_mapbox(
                    self.demand_data,
                    lon="longitude",
                    lat="latitude",
                    color_discrete_sequence=["red"],
                    hover_name="text",
                )
                figure.add_trace(fig2.data[0])

        # Update the layout
        layout = self.layout
        figure.update_layout(**layout)

        return figure

    @property
    def hover_text(self):
        """Return hover text column."""
        if self.demand_data is not None:
            text = (
                self.demand_data["sera_node"]
                + ", "
                + self.demand_data["State"]
                + ". <br>Demand:   "
                + self.demand_data["load"].astype(str)
                + " kg"
            )
        elif self.units == "category":
            try:
                text = (
                    self.df["county"]
                    + " County, "
                    + self.df["state"]
                    + ": <br>   "
                    + self.df[self.color_var].astype(str)
                    + " "
                    + self.units
                )
            except KeyError:
                text = (
                    round(self.df[self.color_var], 2).astype(str)
                    + " "
                    + self.units
                )
        else:
            extra_str = ""
            if "hydrogen_annual_kg" in self.df:
                extra_str += (
                    "<br>    H2 Supply:    "
                    + self.df["hydrogen_annual_kg"].apply(lambda x: f"{x:,}")
                    + " kg    "
                )
            if "dist_to_selected_load" in self.df:
                extra_str += (
                    "<br>    Dist to load:    "
                    + self.df["dist_to_selected_load"].apply(
                        lambda x: f"{x:,.2f}"
                    )
                    + " km    "
                )

            try:
                text = (
                    self.df["county"]
                    + " County, "
                    + self.df["state"]
                    + ":"
                    + extra_str
                    + f"<br>    {convert_to_title(self.color_var)}:   "
                    + self.df[self.color_var].round(2).astype(str)
                    + " "
                    + self.units
                )
            except KeyError:
                text = (
                    extra_str
                    + f"<br>    {convert_to_title(self.color_var)}:   "
                    + self.df[self.color_var].round(2).astype(str)
                    + " "
                    + self.units
                )

        return text

    @property
    def layout(self):
        """Build the map data layout dictionary."""
        layout = copy.deepcopy(MAP_LAYOUT)
        layout["mapbox"]["style"] = self.basemap
        layout["showlegend"] = self.show_legend
        layout["title"]["text"] = self.plot_title
        layout["yaxis"] = dict(range=[self.cmin, self.cmax])
        return layout

    def marker(self, point_size, reverse_color=False):
        """Return marker dictionary."""
        if self.units == "category":
            marker = dict(
                opacity=1.0,
                reversescale=reverse_color,
                size=point_size,
            )
        else:
            marker = dict(
                color=self.df[self.color_var],
                colorscale=COLORS[self.colorscale],
                cmin=None if self.cmin is None else float(self.cmin),
                cmax=None if self.cmax is None else float(self.cmax),
                opacity=1.0,
                reversescale=reverse_color,
                size=point_size,
                colorbar=dict(
                    title=dict(
                        text=self.units,
                        font=dict(
                            size=15, color="white", family="New Times Roman"
                        ),
                    ),
                    tickfont=dict(color="white", family="New Times Roman"),
                ),
            )

        return marker

    @property
    def show_legend(self):
        """Boolean switch to show/hide legend."""
        return self.units == "category"


class ColorRange:
    """Helper class to represent the color range."""

    def __init__(
        self, df, color_var, project=None, color_min=None, color_max=None
    ):
        """Initialize ColorRange object."""
        self.df = df
        self.color_var = color_var
        if project:
            scales = Config(project).scales.get(self.color_var, {})
        else:
            scales = {}
        self._color_min = color_min or scales.get("min")
        self._color_max = color_max or scales.get("max")

    def __iter__(self):
        return iter((self.min, self.max))

    @property
    def min(self):
        """Return appropriate color minimum value."""
        if self._color_max and not self._color_min:
            return self.df[self.color_var].min()

        return self._color_min

    @property
    def max(self):
        """Return appropriate color maximum value."""
        if self._color_min and not self._color_max:
            return self.df[self.color_var].max()

        return self._color_max
