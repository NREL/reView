# -*- coding: utf-8 -*-
"""Element builders.

Methods for building html and core component elements given user inputs in
scenario callbacks.

Created on Fri May 20 12:07:29 2022

@author: twillia2
"""
import json
from collections import Counter

import pandas as pd
import numpy as np
import plotly.express as px

from reView.utils.classes import DiffUnitOptions
from reView.utils.config import Config
from reView.utils.constants import  AGGREGATIONS, DEFAULT_POINT_SIZE
from reView.utils.functions import convert_to_title


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

        # self.aggregations = AGGREGATIONS
        # self.category_check()`

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
            title = [t.replace('$', 'dollars') for t in title]
            title = ['$', r'\Delta', r'\text{'] + title + ['}$']

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

    def binned(self, x, y, bins=100):
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

        # Assign bins as max bin value
        main_df = main_df.dropna(subset=x)
        main_df = main_df.sort_values(x)
        minx, maxx = main_df[x].min(), main_df[x].max()
        xrange = maxx - minx
        bin_size = np.ceil(xrange / bins)
        bins = np.arange(minx, maxx + bin_size, bin_size)

        main_df["xbin"] = bins[-1]
        for bn in bins[::-1]:
            print(bn)
            main_df["xbin"][main_df[x] <= bn] = bn

        main_df["ybin"] = main_df.groupby(["xbin", self.GROUP])[y].transform(
                                "mean"
                          )

        # The simpler line plot part
        main_df = main_df.sort_values([x, self.GROUP])
        agg = AGGREGATIONS.get(
            DiffUnitOptions.remove_from_variable_name(y), "mean"
        )
        yagg = main_df.groupby(["xbin", self.GROUP])[y].transform(agg)
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

    def histogram(self, y, bins=100):
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

        fig = px.histogram(
            main_df,
            x=y,
            labels={y: y_title},
            color=self.GROUP,
            opacity=self.alpha,
            color_discrete_sequence=px.colors.qualitative.Safe,
            barmode="overlay",
            nbins=bins
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
        fig.update_layout(
            font_family="Time New Roman",
            title_font_family="Times New Roman",
            legend_title_font_color="black",
            font_color="white",
            font_size=15,
            margin=dict(l=70, r=20, t=115, b=20),
            hovermode="closest",
            paper_bgcolor="#1663B5",
            legend_title_text=self.GROUP,
            dragmode="select",
            titlefont=dict(color="white", size=18, family="Time New Roman"),
            title=dict(
                text=self.plot_title,
                yref="container",
                x=0.05,
                y=0.94,
                yanchor="bottom",
                pad=dict(b=10),
            ),
            legend=dict(
                title_font_family="Times New Roman",
                bgcolor="#E4ECF6",
                font=dict(family="Times New Roman", size=15, color="black"),
            ),
        )
        if y:
            fig.update_layout(yaxis={"range": self._plot_range(y)})
        return fig