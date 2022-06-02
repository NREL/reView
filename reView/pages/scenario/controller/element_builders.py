# -*- coding: utf-8 -*-
"""Element builders.

Methods for building html and core component elements given user inputs in
scenario callbacks.

Created on Fri May 20 12:07:29 2022

@author: twillia2
"""
import copy
import json
from collections import Counter

import pandas as pd
import numpy as np
import plotly.express as px

from reView.utils.classes import DiffUnitOptions
from reView.utils.config import Config
from reView.utils.constants import DEFAULT_POINT_SIZE, DEFAULT_LAYOUT
from reView.utils.functions import convert_to_title


CHART_LAYOUT = copy.deepcopy(DEFAULT_LAYOUT)
CHART_LAYOUT.update({"legend_title_font_color": "black"})


def is_integer(x):
    """Check if an input is an integer."""
    try:
        int(x)
        check = True
    except ValueError:
        check = False
    return check


class Plots:
    """Class for handling grouped plots."""

    GROUP = "Scenarios"
    DEFAULT_N_BINS = 20

    def __init__(
        self,
        project,
        datasets,
        plot_title,
        point_size=DEFAULT_POINT_SIZE,
        user_scale=(None, None),
        alpha=1,
    ):
        """Initialize plotting object for a reV project."""
        self.datasets = datasets
        self.plot_title = plot_title
        self.point_size = point_size
        self.user_scale = user_scale
        self.alpha = alpha
        self.config = Config(project)

    def __repr__(self):
        """Print representation string."""
        return f"<Plots object: project={self.config.project}>"

    # def category_check(self):
    #     """Check for json dictionary entries and adjust if needed."""
    #     # Use one dataset to check
    #     sample_df = self.datasets[next(iter(self.datasets))]
    #     y = sample_df.columns[1]
    #     if Categories.is_json(sample_df, y):
    #         adjusted_datasets = {}
    #         for key, df in self.datasets.items():
    #             df = df.copy()
    #             df = self.adjust_category(df)
    #             adjusted_datasets[key] = df
    #         self.datasets = adjusted_datasets

    # def adjust_category(self, df):
    #     """Adjust dataset for categorical data."""
    #     # We'll need x and y
    #     x, y = df.columns[:2]
    #     if Categories.is_json(df, y):
    #         df[y] = df[y].apply(json.loads)
    #         df["gid_counts"] = df["gid_counts"].apply(json.loads)

    #     # Find the mode and counts
    #     df["y_mode"] = Categories().mode(df, y)
    #     y_counts = Categories().counts(df, y)
    #     x_portions = {}

    #     # Speed this up?
    #     arg_list = [(df, y, x, k) for k in y_counts]
    #     for args in tqdm(arg_list):
    #         ckey, xp = self._xportions(args)
    #         x_portions[ckey] = xp

    #     # Make the category dataframe
    #     adf = pd.DataFrame({y: y_counts, x: x_portions})
    #     adf["y_mode"] = adf.index

    #     # Append to mode field!
    #     del df[y]
    #     del df[x]
    #     df = pd.merge(df, adf, on="y_mode")

    #     return df

    def _plot_range(self, y):
        """Get plot range."""
        # User defined y-axis limits
        user_ymin, user_ymax = self.user_scale
        scale = self.config.scales.get(
            DiffUnitOptions.remove_from_variable_name(y), {}
        )
        ymin = user_ymin or scale.get("min")
        ymax = user_ymax or scale.get("max")

        if ymin and not ymax:
            ymax = max([df[y].max() for df in self.datasets.values()])
        if ymax and not ymin:
            ymin = min([df[y].min() for df in self.datasets.values()])
        return [ymin, ymax]

    def _axis_title(self, variable):
        """Make a title out of variable name and units."""
        diff = DiffUnitOptions.from_variable_name(variable)
        is_difference = diff is not None
        is_percent_difference = diff == DiffUnitOptions.PERCENTAGE
        variable = DiffUnitOptions.remove_from_variable_name(variable)
        variable = variable.removesuffix("_2")
        title = [self.config.titles.get(variable, convert_to_title(variable))]

        if is_percent_difference:
            title += ["(%)"]
        elif units := self.config.units.get(variable):
            title += [f"({units})"]

        if is_difference:
            # this is a limitation (bug?) of dash...
            # can only have "$" at start and end of string
            title = [t.replace("$", "dollars") for t in title]
            title = ["$", r"\Delta", r"\text{"] + title + ["}$"]

        return " ".join(title)

    def cumulative_sum(self, x, y):
        """Return a cumulative capacity scatterplot."""
        main_df = None
        for key, df in self.datasets.items():
            df = self._fix_doubles(df)
            if main_df is None:
                main_df = df.copy()
                main_df = main_df.sort_values(y)
                main_df["csum"] = main_df[x].cumsum()
                main_df[self.GROUP] = key
            else:
                df = df.sort_values(y)
                df["csum"] = df[x].cumsum()
                df[self.GROUP] = key
                main_df = pd.concat([main_df, df])

        x_title, y_title = self._axis_title(x), self._axis_title(y)
        main_df = main_df.sort_values(self.GROUP)
        # main_df["csum"] = main_df["csum"] / 1_000_000
        fig = px.scatter(
            main_df,
            x="csum",
            y=y,
            custom_data=["sc_point_gid", "print_capacity"],
            labels={
                "csum": f"Cumulative {x_title}",
                y: y_title,
            },
            color=self.GROUP,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )

        fig.update_traces(
            marker=dict(size=self.point_size, line=dict(width=0)),
            unselected=dict(marker=dict(color="grey")),
        )

        return self._update_fig_layout(fig, y)

    def binned(self, x, y, bin_size):
        """Return a line plot."""
        # The clustered scatter plot part
        main_df = None
        for key, df in self.datasets.items():
            df = self._fix_doubles(df)
            if main_df is None:
                df = df.sort_values(x)
                main_df = df.copy()
                main_df[self.GROUP] = key
            else:
                df[self.GROUP] = key
                df = df.sort_values(x)
                main_df = pd.concat([main_df, df])

        main_df["xbin"] = self.assign_bins(main_df[x], bin_size=bin_size)
        main_df["ybin"] = main_df.groupby(["xbin", self.GROUP])[y].transform(
            "mean"
        )

        # The simpler line plot part
        main_df = main_df.sort_values([x, self.GROUP])
        yagg = main_df.groupby(["xbin", self.GROUP])[y].transform("mean")
        main_df["yagg"] = yagg
        line_df = main_df.copy()
        line_df = line_df[["xbin", "yagg", self.GROUP]].drop_duplicates()

        x_title, y_title = self._axis_title(x), self._axis_title(y)

        # Points
        fig = px.scatter(
            main_df,
            x="xbin",
            y="yagg",  # Plot all y's so we can share selections with map
            custom_data=["sc_point_gid", "print_capacity"],
            labels={x: x_title, y: y_title},
            color=self.GROUP,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )

        # Lines
        colors = px.colors.qualitative.Safe
        for i, group in enumerate(line_df[self.GROUP].unique()):
            df = line_df[line_df[self.GROUP] == group]
            lines = px.line(
                df,
                x="xbin",
                y="yagg",
                color=self.GROUP,
                color_discrete_sequence=[colors[i]],
            )  # <---------- We could run out of colors this way
            fig.add_trace(lines.data[0])

        fig.layout["xaxis"]["title"]["text"] = x_title
        fig.layout["yaxis"]["title"]["text"] = y_title

        fig.update_traces(
            marker=dict(size=self.point_size, line=dict(width=0)),
            unselected=dict(marker=dict(color="grey")),
        )

        return self._update_fig_layout(fig, y)

    def scatter(self, x, y):
        """Return a regular scatterplot."""
        main_df = None
        for key, df in self.datasets.items():
            df = self._fix_doubles(df)
            if main_df is None:
                main_df = df.copy()
                main_df[self.GROUP] = key
            else:
                df[self.GROUP] = key
                main_df = pd.concat([main_df, df])

        x_title, y_title = self._axis_title(x), self._axis_title(y)

        main_df = main_df.sort_values(self.GROUP)
        fig = px.scatter(
            main_df,
            x=x,
            y=y,
            opacity=self.alpha,
            custom_data=["sc_point_gid", "print_capacity"],
            labels={x: x_title, y: y_title},
            color=self.GROUP,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )

        fig.update_traces(
            marker_line=dict(width=0),
            marker=dict(size=self.point_size, line=dict(width=0)),
            unselected=dict(marker=dict(color="grey")),
        )

        return self._update_fig_layout(fig, y)

    def histogram(self, y):
        """Return a histogram."""
        main_df = None
        for key, df in self.datasets.items():
            df = self._fix_doubles(df)
            if main_df is None:
                main_df = df.copy()
                main_df[self.GROUP] = key
            else:
                df[self.GROUP] = key
                main_df = pd.concat([main_df, df])

        y_title = self._axis_title(y)
        main_df = main_df.sort_values(self.GROUP)

        # Use preset scales for the x axis and max count for y axis
        # limx = list(self.scales[y].values())

        fig = px.histogram(
            main_df,
            x=y,
            # range_x=limx,
            range_y=[0, 4000],
            labels={y: y_title},
            color=self.GROUP,
            opacity=self.alpha,
            color_discrete_sequence=px.colors.qualitative.Safe,
            barmode="overlay",
        )

        fig.update_traces(
            marker=dict(line=dict(width=0)),
            unselected=dict(marker=dict(color="grey")),
        )

        return self._update_fig_layout(fig, y)

    def char_hist(self, x):
        """Make a histogram of the characterization column."""
        main_df = list(self.datasets.values())[0]
        counts = {}
        for str_dict in main_df[x]:
            if not isinstance(str_dict, str):
                continue
            counts_for_sc_point = json.loads(str_dict)
            for label, count in counts_for_sc_point.items():
                counts[label] = counts.get(label, 0) + count

        labels = sorted(counts, key=lambda k: -counts[k])
        counts = [counts[label] for label in labels]

        data = pd.DataFrame({"Category": labels, "Counts": counts})

        fig = px.bar(
            data,
            x="Category",
            y="Counts",
            labels={
                "Category": self.config.titles.get(x, convert_to_title(x))
            },
            opacity=self.alpha,
            color_discrete_sequence=px.colors.qualitative.Safe,
            barmode="overlay",
        )

        return self._update_fig_layout(fig)

    def box(self, y):
        """Return a boxplot."""

        units = self.config.units.get(
            DiffUnitOptions.remove_from_variable_name(y), ""
        )

        def fix_key(key):
            """Display numbers and strings together."""
            if is_integer(key):
                key = str(key) + units
            return key

        # Infer the y variable and units
        dfs = self.datasets
        df = dfs[list(dfs.keys())[0]]

        main_df = None
        for key, df in dfs.items():
            df = self._fix_doubles(df)
            if main_df is None:
                main_df = df.copy()
                main_df[self.GROUP] = key
            else:
                df[self.GROUP] = key
                main_df = pd.concat([main_df, df])

        y_title = self._axis_title(y)

        if all(main_df[self.GROUP].apply(is_integer)):
            main_df[self.GROUP] = main_df[self.GROUP].astype(int)
        main_df = main_df.sort_values(self.GROUP)
        main_df[self.GROUP] = main_df[self.GROUP].apply(fix_key)

        fig = px.box(
            main_df,
            x=self.GROUP,
            y=y,
            custom_data=["sc_point_gid", "print_capacity"],
            labels={y: y_title},
            color=self.GROUP,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )

        fig.update_traces(
            marker=dict(
                size=self.point_size,
                opacity=1,
                line=dict(
                    width=0,
                ),
            ),
            unselected=dict(marker=dict(color="grey")),
        )

        return self._update_fig_layout(fig, y)

    def bin_boundaries(self, values, bin_size=None):
        """Calculate the bin edges given input values and a bin size.

        Parameters
        ----------
        values : `array_like`
            Input values that will be split into bins. Used to calculate
            the min and max value for bin edges.
        bin_size : float, optional
            Desired width of bins. Can be `None`, which uses the
            `DEFAULT_N_BINS` values set at the class level. If negative,
            will be converted to a positive value.  By default, `None`.

        Returns
        -------
        np.array
            1D array of bin edges. The values start at
            `min(values) - bin_size` and go up to
            `max(values) + bin_size` (inclusive).

        Examples
        --------
        >>> plotter = Plot(...)
        >>> plotter.bin_boundaries(range(60), bin_size=10)
        array([-10,  0,  10,  20,  30,  40,  50,  60,  70])

        >>> assert plotter.DEFAULT_N_BINS == 20
        >>> plotter.bin_boundaries(range(61), bin_size=None)
        array([-3.,  0.,  3.,  6.,  9., 12., ..., 57., 60., 63.])
        """
        min_value, max_value = min(values), max(values)
        max_range = max_value - min_value
        if bin_size is None or abs(bin_size) > max_range:
            bin_size = max_range / self.DEFAULT_N_BINS
        else:
            bin_size = abs(bin_size)
        return np.arange(
            min_value - bin_size, max_value + 2 * bin_size, bin_size
        )

    def assign_bins(self, values, bin_size=None, right=False):
        """Assign bins to inputs.

        This function assigns a `bin` value to each input. The bin value
        represents the left edge of the bin if `right=False`, otherwise
        it represents the right edge of the bin. The edges of the bins
        are determined using the min and max values of the input as
        well as the `bin_size`.

        Parameters
        ----------
        values : `array_like`
            Input values that will be split into bins. Used to calculate
            the min and max value for bin edges. The output assigns a
            bin to each of these values.
        bin_size : float, optional
            Desired width of bins. Can be `None`, which uses the
            `DEFAULT_N_BINS` values set at the class level. If negative,
            will be converted to a positive value.  By default, `None`.
        right : bool, optional
            Option to use the right edges of the bin as the label.
            By default, `False`.

        Returns
        -------
        `array_like`
            An array of bin labels for the input.

        Examples
        --------
        >>> plotter = Plot(...)
        >>> plotter.assign_bins(range(6), bin_size=1, right=False)
        array([1, 2, 3, 4, 5, 6])

        >>> plotter.assign_bins(range(6), bin_size=1, right=True)
        array([0, 1, 2, 3, 4, 5])
        """
        bin_boundaries = self.bin_boundaries(values, bin_size)
        bin_indices = np.digitize(values, bins=bin_boundaries, right=right)
        return bin_boundaries[bin_indices]

    def _fix_doubles(self, df):
        """Check and or fix columns names when they match."""
        if not isinstance(df, pd.core.frame.Series):
            cols = np.array(df.columns)
            counts = Counter(cols)
            for col, count in counts.items():
                if count > 1:
                    idx = np.where(cols == col)[0]
                    cols[idx[1]] = col + "_2"
            df.columns = cols
        return df

    def _update_fig_layout(self, fig, y=None):
        """Update the figure layout with title, etc."""
        layout = copy.deepcopy(CHART_LAYOUT)
        layout["title"]["text"] = self.plot_title
        layout["legend_title_text"] = self.GROUP
        fig.update_layout(**layout)
        if y:
            fig.update_layout(yaxis={"range": self._plot_range(y)})
        return fig
