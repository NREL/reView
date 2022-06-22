# -*- coding: utf-8 -*-
"""Scenario page data model."""
import json
import logging
import multiprocessing as mp
import operator
import os
import platform

from collections import Counter

import numpy as np
import pandas as pd

from pandarallel import pandarallel as pdl
from sklearn.neighbors import BallTree
from sklearn.metrics import DistanceMetric
from tqdm import tqdm

from reView.app import cache, cache2, cache3
from reView.utils.functions import (
    as_float,
    strip_rev_filename_endings,
    lcoe,
    lcot,
    safe_convert_percentage_to_decimal,
    capacity_factor_from_lcoe,
    adjust_cf_for_losses
)
from reView.utils.config import Config

pd.set_option("mode.chained_assignment", None)
pdl.initialize(progress_bar=True, verbose=0)
logger = logging.getLogger(__name__)


DIST_METRIC = DistanceMetric.get_metric("haversine")


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def adjust_capacities(df, project, signal_dict, x_var, chart_selection):
    """Adjust capacities and lcoes for given characterization selection."""
    def adjust_capacity(row, x_var, cats, res, density):
        """Keep given categorical capacity, remove the rest."""
        if not isinstance(row[x_var], float):
            row[x_var] = json.loads(row[x_var])
            if row[x_var]:
                remove = {k: v for k, v in row[x_var].items() if k not in cats}
                removed_cells = sum(remove.values())
                removed_km2 = (res * res * removed_cells) / 1_000_000
                removed_cap = removed_km2 * density
                row["capacity"] -= removed_cap
        return row

    def adjust_lcoe(df, sam, eos):
        """Adjust for economies of scale if possible."""
        # Unpack SAM info
        capex = float(sam["capital_cost"])
        fcr = float(sam["fixed_charge_rate"])
        opex = float(sam["fixed_operating_cost"])

        # Adjust capex, get full cost
        capex *= np.interp(df["capacity"], eos["capacity"], eos["scalar"])
        capex *= df["capacity"] * 1_000

        # Get full opex
        opex *= df["capacity"] * 1_000

        # Calculate generation
        gen = df["mean_cf"] * df["capacity"] * 8_760
        df["mean_lcoe"] = ((capex * fcr) + opex) / gen

        return df

    # Get config
    config = Config(project)

    # Get categories
    cats = [p["label"] for p in chart_selection["points"]]

    # These might've been translated
    if "lookup" in config.characterization_cols[x_var]:
        lookup = config.characterization_cols[x_var]["lookup"]
        ilookup = {v: k for k, v in lookup.items()}
        cats = [ilookup[cat] for cat in cats]

    # It needs to have capacity density for this
    density = config.capacity_density
    res = config.resolution

    # Adjust capacities for each row
    if res and density:
        if platform.system() == "Windows":
            df = df.apply(adjust_capacity, cats=cats, x_var=x_var, res=res,
                          density=density, axis=1)
        else:
            df = df.parallel_apply(adjust_capacity, cats=cats, x_var=x_var,
                                   res=res, density=density, axis=1)

        # Adjust LCOEs if possible
        if config.sam and config.eos:
            # Using pattern recognition for now, in a hurry
            name = config.name_lookup[signal_dict["path"]]
            parts = name.split("_")
            sam = {k: v for k, v in config.sam.items() if k in parts}
            eos = {k: v for k, v in config.eos.items() if k in parts}
            if eos and sam:
                sam = sam[next(iter(sam))]
                eos = eos[next(iter(eos))]
                df = adjust_lcoe(df, sam=sam, eos=eos)

    # Remove cells with no capacity
    df = df[df["capacity"] > 0]

    return df


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def apply_all_selections(df, signal_dict, map_function, project,
                         chart_selection, map_selection, click_selection,
                         y_var, x_var, chart_type):
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
    # demand_data = None

    # # If there is a selection in the chart, filter these points
    # if map_function == "demand":
    #     demand_data = Config(project).demand_data
    #     demand_data["load"] = demand_data["H2_MT"] * 1e3  # convert to kg
    #     if click_selection and len(click_selection["points"]) > 0:
    #         point = click_selection["points"][0]
    #         if point["curveNumber"] == 1:
    #             df, demand_data = filter_on_load_selection(
    #                 df, point["pointIndex"], demand_data
    #             )
    #     elif map_selection and len(map_selection["points"]) > 0:
    #         selected_demand_points = [
    #             p for p in map_selection["points"] if p["curveNumber"] == 1
    #         ]
    #         if not selected_demand_points:
    #             mean_lat = np.mean([p["lat"] for p in map_selection["points"]])
    #             mean_lon = np.mean([p["lon"] for p in map_selection["points"]])
    #             selection_coords = np.radians([[mean_lat, mean_lon]])
    #             load_center_ind = closest_demand_to_coords(
    #                 selection_coords, demand_data
    #             )
    #             df, demand_data = filter_on_load_selection(
    #                 df, load_center_ind, demand_data
    #             )
    #         else:
    #             df["demand_connect_count"] = 0
    #             demand_idxs = []
    #             for point in selected_demand_points:
    #                 demand_idxs.append(point["pointIndex"])
    #                 selected_df, __ = filter_on_load_selection(
    #                     df, point["pointIndex"], demand_data
    #                 )
    #                 df.loc[
    #                     df.sc_point_gid.isin(selected_df.sc_point_gid),
    #                     "demand_connect_count",
    #                 ] += 1
    #             df = df[df["demand_connect_count"] > 0]
    #             demand_data = demand_data.iloc[demand_idxs]

    # elif map_function == "meet_demand":
    #     demand_data = Config(project).demand_data
    #     demand_data["load"] = demand_data["H2_MT"] * 1e3  # convert to kg

    #     # add h2 data
    #     demand_coords = demand_data[["latitude", "longitude"]].values
    #     sc_coords = df[["latitude", "longitude"]].values
    #     demand_coords = np.radians(demand_coords)
    #     sc_coords = np.radians(sc_coords)
    #     tree = BallTree(demand_coords, metric="haversine")
    #     __, ind = tree.query(sc_coords, return_distance=True, k=1)
    #     df["h2_load_id"] = demand_data["OBJECTID"].values[ind]
    #     filtered_points = []
    #     for d_id in df["h2_load_id"].unique():
    #         temp_df = df[df["h2_load_id"] == d_id].copy()
    #         temp_df = temp_df.sort_values("total_lcoh_fcr")
    #         temp_df["h2_supply"] = temp_df["hydrogen_annual_kg"].cumsum()
    #         load = demand_data[demand_data["OBJECTID"] == d_id]["load"].iloc[0]
    #         where_inds = np.where(temp_df["h2_supply"] <= load)[0]
    #         if where_inds.size > 0:
    #             final_ind = where_inds.max() + 1
    #             filtered_points.append(temp_df.iloc[0:final_ind])
    #         else:
    #             filtered_points.append(temp_df)
    #     df = pd.concat(filtered_points)
    #     demand_data = demand_data[
    #         demand_data["OBJECTID"].isin(df["h2_load_id"].unique())
    #     ]

    # else:
    # If there is a selection in the map, filter these points
    if map_selection and len(map_selection["points"]) > 0:
        df = point_filter(df, map_selection)

    if chart_selection and len(chart_selection["points"]) > 0:
        if chart_type == "char_histogram":
            if y_var.endswith("_mode"):
                categories = [p["label"] for p in chart_selection["points"]]
                df = df[df[y_var].isin(categories)]
            elif not signal_dict["path2"]:
                df = adjust_capacities(df, project, signal_dict, x_var,
                                       chart_selection)
        elif chart_type == "histogram":
            points = chart_selection["points"]
            bin_size = points[0]["customdata"][0]
            sdfs = []
            for point in points:
                bottom_bin = point["x"]
                top_bin = bottom_bin + bin_size
                sdf = df[(df[y_var] >= bottom_bin) & (df[y_var] < top_bin)]
                sdfs.append(sdf)
            df = pd.concat(sdfs)
        else:
            df = point_filter(df, chart_selection)

    return df, None


def apply_filters(df, filters):
    """Apply filters from string entries to dataframe."""

    ops = {
        ">=": operator.ge,
        ">": operator.gt,
        "<=": operator.le,
        "<": operator.lt,
        "==": operator.eq,
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


def calc_mask(df1, df2, unique_id_col="sc_point_gid"):
    """Remove the areas in df2 that are in df1."""
    # How to deal with mismatching grids?
    df = df2[~df2[unique_id_col].isin(df1[unique_id_col])]
    return df


def least_cost(dfs, bycol="total_lcoe", group_col="sc_point_gid"):
    """Return a single least cost df from a list dfs."""
    # Make one big data frame
    bdf = pd.concat(dfs)
    bdf = bdf.reset_index(drop=True)

    # Group, find minimum, and subset
    idx = bdf.groupby(group_col)[bycol].idxmin()
    data = bdf.iloc[idx]

    return data


def read_df_and_store_scenario_name(file):
    """Retrieve a single data frame."""
    data = pd.read_csv(file, low_memory=False)
    data["scenario"] = strip_rev_filename_endings(file.name)
    return data


def calc_least_cost(paths, out_file, bycol="total_lcoe"):
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
        data = least_cost(dfs, bycol=bycol)
        data.to_csv(out_file, index=False)


def key_mode(dct):
    """Return the key associated with the modal value in a dictionary."""
    if dct:
        value = Counter(dct).most_common()[0][0]
    else:
        print(dct)
        value = "nan"
    return value


# pylint: disable=no-member
# pylint: disable=unsubscriptable-object
# pylint: disable=unsupported-assignment-operation
@cache.memoize()
def cache_table(project, path, y_var, x_var, recalc_table=None, recalc="off"):
    """Read in just a single table."""
    # Get config
    config = Config(project)
    all_cols = pd.read_csv(path, nrows=0).columns

    # Get the table
    if recalc == "on":
        data = ReCalculatedData(config=Config(project)).build(
            path, recalc_table
        )
    else:
        cols = [y_var, x_var, "mean_cf", "capacity", "area_sq_km",
                "sc_point_gid", "state", "county", "latitude", "longitude"]
        if "nrel_region" in all_cols:
            cols.append("nrel_region")

        data = pd.read_csv(path, usecols=cols, low_memory=False)

    # We want some consistent fields
    if "capacity" not in data.columns and "hybrid_capacity" in data.columns:
        data["capacity"] = data["hybrid_capacity"].copy()
    if "print_capacity" not in data.columns:
        data["print_capacity"] = data["capacity"].copy()

    # If characterization, use modal category
    if x_var in config.characterization_cols:
        ncol = x_var + "_mode"
        cdata = data[(~data[y_var].isnull()) & (data[y_var] != "{}")]
        odata = data[(data[y_var].isnull()) | (data[y_var] == "{}")]
        odata[ncol] = "nan"
        cdata[ncol] = cdata[x_var].map(json.loads)
        cdata[ncol] = cdata[ncol].apply(key_mode)
        if "lookup" in config.characterization_cols[x_var]:
            lookup = config.characterization_cols[x_var]["lookup"]
            cdata[ncol] = cdata[ncol].map(lookup)
        data = pd.concat([odata, cdata])

    # If characterization, use modal category
    if y_var in config.characterization_cols:
        ncol = y_var + "_mode"
        cdata = data[(~data[y_var].isnull()) & (data[y_var] != "{}")]
        odata = data[(data[y_var].isnull()) | (data[y_var] == "{}")]
        odata[ncol] = "nan"
        cdata[ncol] = cdata[x_var].map(json.loads)
        cdata[ncol] = cdata[ncol].apply(key_mode)
        if "lookup" in config.characterization_cols[y_var]:
            lookup = config.characterization_cols[y_var]["lookup"]
            cdata[ncol] = cdata[ncol].astype(float).astype(int).astype(str)
            cdata[ncol] = cdata[ncol].map(lookup)
        data = pd.concat([odata, cdata])

    return data


@cache2.memoize()
def cache_map_data(signal_dict):
    """Read and store a data frame from the config and options given."""
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
    diff_units = signal_dict["diff_units"]
    y_var = signal_dict["y"]
    x_var = signal_dict["x"]

    # Unpack recalc table
    recalc_a = recalc_tables["scenario_a"]
    recalc_b = recalc_tables["scenario_b"]

    # Read and cache first table
    df1 = cache_table(project, path, y_var, x_var, recalc_a, recalc)

    # Apply filters
    df1 = apply_filters(df1, filters)

    # If there's a second table, read/cache the difference
    if path2 and os.path.isfile(path2):
        # Match the format of the first dataframe
        df2 = cache_table(project, path2, y_var, x_var, recalc_b, recalc)
        df2 = apply_filters(df2, filters)

        # If the difference option is specified difference
        calculator = Difference(
            index_col="sc_point_gid",
            diff_units=diff_units
        )
        df = calculator.calc(df1, df2, y_var)

        # If mask, try that here
        if mask == "on":
            df = calc_mask(df1, df)
    else:
        df = df1

    # Filter for states
    if states:
        if any(df["state"].isin(states)):
            df = df[df["state"].isin(states)]

        if "offshore" in states:
            df = df[df["offshore"] == 1]
        if "onshore" in states:
            df = df[df["offshore"] == 0]

    # Filter for regions
    if regions:
        if any(df["nrel_region"].isin(regions)):
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
            if any(df["state"].isin(states)):
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
    """_summary_

    Parameters
    ----------
    df : _type_
        _description_
    load_center_ind : _type_
        _description_
    demand_data : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    load_center_coords, load = closest_load_center(
        load_center_ind, demand_data
    )
    demand_data = demand_data.iloc[load_center_ind: load_center_ind + 1]
    df = filter_points_by_demand(df, load_center_coords, load)
    return df, demand_data


def closest_load_center(load_center_ind, demand_data):
    """_summary_

    Parameters
    ----------
    load_center_ind : _type_
        _description_
    demand_data : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    demand_coords = demand_data[["latitude", "longitude"]].values
    demand_coords_rad = np.radians(demand_coords)
    load_center_info = demand_data.iloc[load_center_ind]
    load_center_coords = demand_coords_rad[load_center_ind]
    load = load_center_info[["load"]].values[0]
    return load_center_coords, load


def filter_points_by_demand(df, load_center_coords, load):
    """_summary_

    Parameters
    ----------
    df : _type_
        _description_
    load_center_coords : _type_
        _description_
    load : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    sc_coords = df[["latitude", "longitude"]].values
    sc_coords = np.radians(sc_coords)
    load_center_coords = np.array(load_center_coords).reshape(-1, 2)
    out = DIST_METRIC.pairwise(load_center_coords, sc_coords)
    # print(out.shape, df.shape)
    df["dist_to_selected_load"] = out.reshape(-1) * 6373.0
    df["selected_load_pipe_lcoh_component"] = (
        df["pipe_lcoh_component"]
        / df["dist_to_h2_load_km"]
        * df["dist_to_selected_load"]
    )
    df["selected_lcoh"] = (
        df["no_pipe_lcoh_fcr"] + df["selected_load_pipe_lcoh_component"]
    )
    df = df.sort_values("selected_lcoh")
    df["h2_supply"] = df["hydrogen_annual_kg"].cumsum()
    where_inds = np.where(df["h2_supply"] >= load)[0]
    # print(f'{load=}')
    # max_supply = df["h2_supply"].max()
    # print(f'{max_supply=}')
    # print(f'{where_inds=}')
    if where_inds.size > 0:
        final_ind = np.where(df["h2_supply"] >= load)[0].min() + 1
        df = df.iloc[0:final_ind]
    # print(f'{df=}')
    return df


def closest_demand_to_coords(selection_coords, demand_data):
    """_summary_

    Parameters
    ----------
    selection_coords : _type_
        _description_
    demand_data : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    demand_coords = demand_data[["latitude", "longitude"]].values
    demand_coords_rad = np.radians(demand_coords)
    out = DIST_METRIC.pairwise(np.r_[selection_coords, demand_coords_rad])
    load_center_ind = np.argmin(out[0][1:])
    return load_center_ind


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

    def __init__(self, index_col="sc_point_gid", diff_units=False):
        """Initialize Difference object."""
        self.index_col = index_col
        self.diff_units = diff_units

    def calc(self, df1, df2, y_var):
        """Calculate difference between each row in two data frames."""
        logger.debug("Calculating difference...")

        # Set index
        df1 = df1.set_index(self.index_col, drop=False)
        df2 = df2.set_index(self.index_col, drop=False)

        # Filter for variable
        df1 = df1.dropna(subset=y_var)
        df2 = df2.dropna(subset=y_var)
        df1 = df1.drop_duplicates(subset="sc_point_gid")
        df2 = df2.drop_duplicates(subset="sc_point_gid")

        # Find common index positions
        idx = list(set(df2[self.index_col]).intersection(df1[self.index_col]))
        df1 = df1.loc[idx]
        df2 = df2.loc[idx]

        # Calculate difference
        df = self.difference(df1, df2, y_var)
        logger.debug("Difference calculated.")

        return df

    def difference(self, df1, df2, y_var):
        """Return single dataset with difference between two."""
        diff = df1[y_var] - df2[y_var]
        if self.diff_units == "percent":
            diff = (diff / df1[y_var]) * 100
            col = f"{y_var}_difference_percent"
        else:
            col = f"{y_var}_difference"
        df1[col] = diff
        return df1


class ReCalculatedData:
    """Class to handle data access and recalculations."""

    def __init__(self, config):
        """Initialize Data object."""
        self.config = config

    def build(self, path, recalc_table=None):
        """Read in a data table given a scenario with re-calc.

        Parameters
        ----------
        path : str
            The scenario data path for the desired data table.
        recalc_table : dict
            A dictionary of parameter-value pairs needed to recalculate
            variables.

        Returns
        -------
        `pd.core.frame.DataFrame`
            A supply-curve data frame with either the original values or
            recalculated values if new parameters are given.
        """
        # This can be a path or a scenario
        data = pd.read_csv(path, low_memory=False)

        # Recalculate if needed, else return original table
        if isinstance(recalc_table, str):
            recalc_table = json.loads(recalc_table)
        if any(recalc_table.values()):
            data = self.re_calc(data, path, recalc_table)

        return data

    def re_calc(self, data, path, recalc_table):
        """Recalculate LCOE for a data frame given a specific FCR.

        Parameters
        ----------
        scenario : str
            The scenario key for the desired data table.
        recalc_table : dict
            A dictionary of parameter-value pairs needed to recalculate
            variables.

        Returns
        -------
        pd.core.frame.DataFrame
            A supply-curve module data frame with recalculated values.
        """
        # If any of these aren't specified, use the original values
        scenario = self.path_lookup[path]
        ovalues = self.original_parameters(scenario)
        for key, value in recalc_table.items():
            if not value:
                recalc_table[key] = ovalues[key]
            else:
                recalc_table[key] = as_float(recalc_table[key])

        # Get the right units for percentages
        ovalues["fcr"] = safe_convert_percentage_to_decimal(ovalues["fcr"])
        recalc_table["fcr"] = safe_convert_percentage_to_decimal(
            recalc_table["fcr"]
        )
        original_losses = safe_convert_percentage_to_decimal(ovalues["losses"])
        new_losses = safe_convert_percentage_to_decimal(recalc_table["losses"])

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
        data["mean_lcoe"] = lcoe(capacity, mean_cf_adj, recalc_table)
        data["lcot"] = lcot(capacity, trans_cap_cost, mean_cf, recalc_table)
        data["total_lcoe"] = data["mean_lcoe"] + data["lcot"]

        return data

    def original_parameters(self, scenario):
        """Return the original parameters for fcr, capex, opex, and losses."""
        fields = self._find_fields(scenario)
        params = self.config.parameters[scenario]
        ovalues = {}
        for key in ["fcr", "capex", "opex", "losses"]:
            ovalues[key] = as_float(params[fields[key]])
        return ovalues

    @property
    def path_lookup(self):
        """Return a path to scenario name lookup."""
        return {str(val): key for key, val in self.config.files.items()}

    def _find_fields(self, scenario):
        """Find input fields with pattern recognition."""
        params = self.config.parameters[scenario]
        patterns = {k.lower().replace(" ", ""): k for k in params.keys()}
        matches = {}
        for key in ["capex", "opex", "fcr", "losses"]:
            match = [v for k, v in patterns.items() if key in str(k)][0]
            matches[key] = match
        return matches
