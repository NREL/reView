# -*- coding: utf-8 -*-
"""Scenario page data model."""
import os
import json
import logging
import operator
import multiprocessing as mp
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.neighbors import BallTree
from tqdm import tqdm

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
from reView.app import cache, cache2, cache3

pd.set_option("mode.chained_assignment", None)
logger = logging.getLogger(__name__)


def apply_all_selections(df, map_func, project, chartsel, mapsel, clicksel):
    """_summary_

    Parameters
    ----------
    df : _type_
        _description_
    map_func : _type_
        _description_
    project : _type_
        _description_
    chartsel : _type_
        _description_
    mapsel : _type_
        _description_
    clicksel : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    demand_data = None

    # If there is a selection in the chart, filter these points
    if map_func == "demand":
        demand_data = Config(project).demand_data
        demand_data["load"] = demand_data["H2_MT"] * 1e3  # convert to kg
        if clicksel and len(clicksel["points"]) > 0:
            point = clicksel["points"][0]
            if point["curveNumber"] == 1:
                df, demand_data = filter_on_load_selection(
                    df, point["pointIndex"], demand_data
                )
        elif mapsel and len(mapsel["points"]) > 0:
            selected_demand_points = [
                p for p in mapsel["points"] if p["curveNumber"] == 1
            ]
            if not selected_demand_points:
                mean_lat = np.mean([p["lat"] for p in mapsel["points"]])
                mean_lon = np.mean([p["lon"] for p in mapsel["points"]])
                selection_coords = np.radians([[mean_lat, mean_lon]])
                load_center_ind = closest_demand_to_coords(
                    selection_coords, demand_data
                )
                df, demand_data = filter_on_load_selection(
                    df, load_center_ind, demand_data
                )
            else:
                df["demand_connect_count"] = 0
                demand_idxs = []
                for point in selected_demand_points:
                    demand_idxs.append(point["pointIndex"])
                    selected_df, __ = filter_on_load_selection(
                        df, point["pointIndex"], demand_data
                    )
                    df.loc[
                        df.sc_point_gid.isin(selected_df.sc_point_gid),
                        "demand_connect_count",
                    ] += 1
                df = df[df["demand_connect_count"] > 0]
                demand_data = demand_data.iloc[demand_idxs]

    elif map_func == "meet_demand":
        demand_data = Config(project).demand_data
        demand_data["load"] = demand_data["H2_MT"] * 1e3  # convert to kg

        # add h2 data
        demand_coords = demand_data[["latitude", "longitude"]].values
        sc_coords = df[["latitude", "longitude"]].values
        demand_coords = np.radians(demand_coords)
        sc_coords = np.radians(sc_coords)
        tree = BallTree(demand_coords, metric="haversine")
        __, ind = tree.query(sc_coords, return_distance=True, k=1)
        df["h2_load_id"] = demand_data["OBJECTID"].values[ind]
        filtered_points = []
        for d_id in df["h2_load_id"].unique():
            temp_df = df[df["h2_load_id"] == d_id].copy()
            temp_df = temp_df.sort_values("total_lcoh_fcr")
            temp_df["h2_supply"] = temp_df["hydrogen_annual_kg"].cumsum()
            load = demand_data[demand_data["OBJECTID"] == d_id]["load"].iloc[0]
            where_inds = np.where(temp_df["h2_supply"] <= load)[0]
            if where_inds.size > 0:
                final_ind = where_inds.max() + 1
                filtered_points.append(temp_df.iloc[0:final_ind])
            else:
                filtered_points.append(temp_df)
        df = pd.concat(filtered_points)
        demand_data = demand_data[
            demand_data["OBJECTID"].isin(df["h2_load_id"].unique())
        ]

    else:
        # If there is a selection in the map, filter these points
        if mapsel and len(mapsel["points"]) > 0:
            df = point_filter(df, mapsel)

    if chartsel and len(chartsel["points"]) > 0:
        df = point_filter(df, chartsel)

    # print(f"{df.columns=}")

    return df, demand_data


def apply_filters(df, filters):
    """Apply filters from string entries to dataframe."""

    ops = {
        ">=": operator.ge,
        ">": operator.gt,
        "<=": operator.le,
        "<": operator.lt,
        "==": operator.eq
    }

    for filter_ in filters:
        if filter_:
            var, operator_, value = filter_.split()
            if var in df.columns:
                operator_ = ops[operator_]
                value = float(value)
                df = df[operator_(df[var], value)]

    return df


def build_name(path):
    """Infer scenario name from path."""
    file = os.path.basename(path)
    name = strip_rev_filename_endings(file)
    name = " ".join([n.capitalize() for n in name.split("_")])
    return name


def point_filter(df, selection):
    """Filter a dataframe by points selected from the chart."""
    if selection:
        points = selection["points"]
        gids = [p.get("customdata", [None])[0] for p in points]
        df = df[df["sc_point_gid"].isin(gids)]
    return df


def is_integer(x):
    """Check if an input is an integer."""
    try:
        int(x)
        check = True
    except ValueError:
        check = False
    return check


def calc_mask(df1, df2, unique_id_col="sc_point_gid"):
    """Remove the areas in df2 that are in df1."""
    # How to deal with mismatching grids?
    df = df2[~df2[unique_id_col].isin(df1[unique_id_col])]
    return df


def least_cost(dfs, by="total_lcoe", group_col="sc_point_gid"):
    """Return a single least cost df from a list dfs."""
    # Make one big data frame
    bdf = pd.concat(dfs)
    bdf = bdf.reset_index(drop=True)

    # Group, find minimum, and subset
    idx = bdf.groupby(group_col)[by].idxmin()
    data = bdf.iloc[idx]

    return data


def read_df_and_store_scenario_name(file):
    """Retrieve a single data frame."""
    data = pd.read_csv(file, low_memory=False)
    data["scenario"] = strip_rev_filename_endings(file.name)
    return data


def calc_least_cost(paths, out_file, by="total_lcoe"):
    """Build the single least cost table from a list of tables."""
    # Not including an overwrite option for now
    if not os.path.exists(out_file):

        # Collect all data frames - biggest lift of all
        paths.sort()
        dfs = []
        with mp.Pool(10) as pool:
            for data in tqdm(
                pool.imap(read_df_and_store_scenario_name, paths),
                total=len(paths),
            ):
                dfs.append(data)

        # Make one big data frame and save
        data = least_cost(dfs, by=by)
        data.to_csv(out_file, index=False)


# pylint: disable=no-member
# pylint: disable=unsubscriptable-object
# pylint: disable=unsupported-assignment-operation
@cache.memoize()
def cache_table(project, path, recalc_table=None, recalc="off"):
    """Read in just a single table."""
    # Get the table
    if recalc == "on":
        data = ReCalculatedData(config=Config(project)).build(
            path, recalc_table
        )
    else:
        data = pd.read_csv(path)

    # We want some consistent fields
    data["index"] = data.index
    if "capacity" not in data.columns and "hybrid_capacity" in data.columns:
        data["capacity"] = data["hybrid_capacity"].copy()
    if "print_capacity" not in data.columns:
        data["print_capacity"] = data["capacity"].copy()
    return data


@cache2.memoize()
def cache_map_data(signal_dict):
    """Read and store a data frame from the config and options given."""
    # Save arguments for later

    # Get signal elements
    filters = signal_dict["filters"]
    mask = signal_dict["mask"]
    path = signal_dict["path"]
    path2 = signal_dict["path2"]
    project = signal_dict["project"]
    recalc_tables = signal_dict["recalc_table"]
    recalc = signal_dict["recalc"]
    states = signal_dict["states"]
    regions = signal_dict["regions"]
    config = Config(project)

    # Unpack recalc table
    recalc_a = recalc_tables["scenario_a"]
    recalc_b = recalc_tables["scenario_b"]

    # Read and cache first table
    df1 = cache_table(project, path, recalc_a, recalc)

    # Apply filters
    df1 = apply_filters(df1, filters)

    # For other functions this data frame needs an x field
    # if y == x:
    #     df1 = df1.iloc[:, 1:]

    # If there's a second table, read/cache the difference
    if path2:
        # Match the format of the first dataframe
        df2 = cache_table(project, path2, recalc_b, recalc)
        df2 = apply_filters(df2, filters)
        # df2 = df2[keepers]

        # if y == x:
        #     df2 = df2.iloc[:, 1:]

        # If the difference option is specified difference
        if DiffUnitOptions.from_variable_name(signal_dict["y"]) is not None:
            # Save for later  <------------------------------------------------ How should we handle this? Optional save button...perhaps a clear saved datasets button?
            target_dir = config.directory / ".review"
            target_dir.mkdir(parents=True, exist_ok=True)

            s1 = os.path.basename(path).replace("_sc.csv", "")
            s2 = os.path.basename(path2).replace("_sc.csv", "")

            fpath = f"diff_{s1}_vs_{s2}_sc.csv"
            dst = target_dir / fpath

            # If we haven't build this build it
            if not dst.exists() or filters:
                calculator = Difference(index_col='sc_point_gid')
                df = calculator.calc(df1, df2)
                if not filters:
                    df.to_csv(dst, index=False)
            else:
                df = pd.read_csv(dst)

            # TODO: The two lines below might honestly be faster... I/O is SLOW
            # calculator = Difference(index_col='sc_point_gid')
            # df = calculator.calc(df1, df2)
        else:
            df = df2.copy()

        # If mask, try that here
        if mask == "on":
            df = calc_mask(df1, df)
    else:
        df = df1.copy()

    # Filter for states
    if states:
        if any(s in df["state"].values for s in states):
            df = df[df["state"].isin(states)]

        if "offshore" in states:
            df = df[df["offshore"] == 1]
        if "onshore" in states:
            df = df[df["offshore"] == 0]

    # Filter for regions
    if regions:
        if any([s in df["nrel_region"].values for s in regions]):
            df = df[df["nrel_region"].isin(regions)]

    return df


@cache3.memoize()
def cache_chart_tables(
    signal_dict,
    region="national",
    # idx=None
):
    """Read and store a data frame from the config and options given."""
    signal_copy = signal_dict.copy()

    # Unpack subsetting information
    states = signal_copy["states"]
    # x = signal_copy["x"]
    # y = signal_copy["y"]

    # If multiple tables selected, make a list of those files
    if signal_copy["added_scenarios"]:
        files = [signal_copy["path"]] + signal_copy["added_scenarios"]
    else:
        files = [signal_copy["path"]]

    # Remove additional scenarios from signal_dict for the cache's sake
    del signal_copy["added_scenarios"]

    # Make a signal copy for each file
    signal_dicts = []
    for file in files:
        signal = signal_copy.copy()
        signal["path"] = file
        signal_dicts.append(signal)

    # Get the requested data frames
    dfs = {}
    for signal in signal_dicts:
        name = build_name(signal["path"])
        df = cache_map_data(signal)
        # first_cols = [x, y, "state", "sc_point_gid"]
        # rest_of_cols = set(df.columns) - set(first_cols)
        # all_cols = first_cols + sorted(rest_of_cols)
        # df = df[all_cols]
        # TODO: Where to add "nrel_region" col?

        # Subset by index selection
        # if idx:
        #     df = df.iloc[idx]

        # Subset by state selection
        if states:
            if any([s in df["state"] for s in states]):
                df = df[df["state"].isin(states)]

            if "offshore" in states:
                df = df[df["offshore"] == 1]
            if "onshore" in states:
                df = df[df["offshore"] == 0]

        # Divide into regions if one table (cancel otherwise for now)
        if region != "national" and len(signal_dicts) == 1:
            regions = df[region].unique()
            dfs = {r: df[df[region] == r] for r in regions}
        else:
            dfs[name] = df

    return dfs

def filter_on_load_selection(df, load_center_ind, demand_data):
    load_center_coords, load = closest_load_center(
        load_center_ind, demand_data
    )
    demand_data = demand_data.iloc[load_center_ind : load_center_ind + 1]
    df = filter_points_by_demand(df, load_center_coords, load)
    return df, demand_data



# def add_new_option(new_option, options):
#     """Add a new option to the options list, replacing an old one if needed.

#     Parameters
#     ----------
#     new_option : dict
#         Option dict with 'label' and 'value' keys.
#     options : list
#         List of existing option dictionaries, in the same format
#         as required of `new_option`.

#     Returns
#     -------
#     list
#         List of option dictionaries including the new option,
#         either appended or replacing an old one.
#     """
#     options = [o for o in options if o["label"] != new_option["label"]]
#     options += [new_option]
#     return options


class Difference:
    """Class to handle supply curve difference calculations."""

    def __init__(self, index_col):
        """Initialize Difference object."""
        self.index_col = index_col

    def calc(self, df1, df2):
        """Calculate difference between each row in two data frames."""
        logger.debug("Calculating difference...")

        df1 = df1.set_index(self.index_col, drop=False)
        df2 = df2.set_index(self.index_col, drop=False)

        common_columns = common_numeric_columns(df1, df2)
        difference = df2[common_columns] - df1[common_columns]
        pct_difference = (difference / df1[common_columns]) * 100
        df1 = df1.merge(
            difference, how='left',
            left_index=True, right_index=True,
            suffixes=('', DiffUnitOptions.ORIGINAL)
        )
        df1 = df1.merge(
            pct_difference, how='left',
            left_index=True, right_index=True,
            suffixes=('', DiffUnitOptions.PERCENTAGE)
        )

        logger.debug("Difference calculated.")
        return df1


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
                if np.isnan(str_dict):
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


class ReCalculatedData:
    """Class to handle data access and recalculations."""

    def __init__(self, config):
        """Initialize Data object."""
        self.config = config

    def build(self, scenario, re_calcs=None):
        """Read in a data table given a scenario with re-calc.

        Parameters
        ----------
        scenario : str
            The scenario key or data path for the desired data table.
        fcr : str | numeric
            Fixed charge as a percentage.
        capex : str | numeric
            Capital expenditure in USD / KW
        opex : str | numeric
            Fixed operating costs in USD / KW
        losses : str | numeric
            Generation losses as a percentage.

        Returns
        -------
        `pd.core.frame.DataFrame`
            A supply-curve data frame with either the original values or
            recalculated values if new parameters are given.
        """
        # This can be a path or a scenario
        scenario = strip_rev_filename_endings(scenario)

        data = self.read(scenario)

        # Recalculate if needed, else return original table
        if any(re_calcs.values()):
            data = self.re_calc(data, scenario, re_calcs)

        return data

    def read(self, path):
        """Read in the needed columns of a supply-curve csv.

        Parameters
        ----------
        scenario : str
            The scenario key for the desired data table.

        Returns
        -------
        `pd.core.frame.DataFrame`
            A supply-curve table with original values.
        """
        # Find the path and columns associated with this scenario
        if not os.path.isfile(path):
            path = self.config.files[path]
        return pd.read_csv(path, low_memory=False)

    def re_calc(self, data, scenario, re_calcs):
        """Recalculate LCOE for a data frame given a specific FCR.

        Parameters
        ----------
        scenario : str
            The scenario key for the desired data table.
        re_calcs : dict
            A dictionary of parameter-value pairs needed to recalculate
            variables.

        Returns
        -------
        pd.core.frame.DataFrame
            A supply-curve module data frame with recalculated values.
        """

        # If any of these aren't specified, use the original values
        ovalues = self.original_parameters(scenario)
        for key, value in re_calcs.items():
            if not value:
                re_calcs[key] = ovalues[key]
            else:
                re_calcs[key] = as_float(re_calcs[key])

        # Get the right units for percentages
        ovalues["fcr"] = safe_convert_percentage_to_decimal(ovalues["fcr"])
        re_calcs["fcr"] = safe_convert_percentage_to_decimal(re_calcs["fcr"])
        original_losses = safe_convert_percentage_to_decimal(ovalues["losses"])
        new_losses = safe_convert_percentage_to_decimal(re_calcs["losses"])

        # Extract needed variables as vectors
        capacity = data["capacity"].values
        mean_cf = data["mean_cf"].values
        mean_lcoe = data["mean_lcoe"].values
        trans_cap_cost = data["trans_cap_cost"].values

        # Adjust capacity factor for LCOE
        mean_cf_adj = capacity_factor_from_lcoe(capacity, mean_lcoe, ovalues)
        mean_cf_adj = adjust_cf_for_losses(
            mean_cf_adj, new_losses, original_losses
        )
        mean_cf = adjust_cf_for_losses(mean_cf, new_losses, original_losses)

        # Recalculate figures
        data["mean_cf"] = mean_cf  # What else will this affect?
        data["mean_lcoe"] = lcoe(capacity, mean_cf_adj, re_calcs)
        data["lcot"] = lcot(capacity, trans_cap_cost, mean_cf, re_calcs)
        data["total_lcoe"] = data["mean_lcoe"] + data["lcot"]

        return data

    def original_parameters(self, scenario):
        """Return the original parameters for fcr, capex, opex, and losses."""
        fields = self._find_fields(scenario)
        params = self.config.parameters[scenario]
        ovalues = dict()
        for key in ["fcr", "capex", "opex", "losses"]:
            ovalues[key] = as_float(params[fields[key]])
        return ovalues

    def _find_fields(self, scenario):
        """Find input fields with pattern recognition."""
        params = self.config.parameters[scenario]
        patterns = {k.lower().replace(" ", ""): k for k in params.keys()}
        matches = {}
        for key in ["capex", "opex", "fcr", "losses"]:
            match = [v for k, v in patterns.items() if key in str(k)][0]
            matches[key] = match
        return matches
