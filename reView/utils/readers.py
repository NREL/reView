"""reView file reading functions."""
import datetime as dt
import json
import logging
import multiprocessing as mp
import os

from pathlib import Path

import h5py
import pandas as pd
import pyarrow as pa

from pyarrow.parquet import ParquetFile
from tqdm import tqdm
from xlrd import XLRDError

from reView.app import cache, cache2, cache3, cache4
from reView.layout.options import REGIONS
from reView.utils.config import Config
from reView.utils.functions import decode, strip_rev_filename_endings

from reView.pages.rev.model import (
    apply_filters,
    calc_mask,
    composite,
    key_mode,
    point_filter,
    Difference,
    ReCalculatedData
)

logger = logging.getLogger(__name__)


TIME_PATTERN = "%Y-%m-%d %H:%M:%S+00:00"


def build_name(path):
    """Infer scenario name from path."""
    file = os.path.basename(path)
    name = strip_rev_filename_endings(file)
    name = " ".join([n.capitalize() for n in name.split("_")])
    return name


@cache.memoize()
def cache_chart_tables(signal_dict):
    """Read and store all dataframes for the chart element."""
    # Unpack subsetting information
    signal_copy = signal_dict.copy()
    states = signal_copy["states"]
    regions = signal_dict["regions"]

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

        # Subset by state selection
        if states:
            if any(df["state"].isin(states)):
                df = df[df["state"].isin(states)]

            if "offshore" in states:
                df = df[df["offshore"] == 1]
            if "onshore" in states:
                df = df[df["offshore"] == 0]

        # Divide into regions if one table (cancel otherwise for now)
        if regions is not None and len(signal_dicts) == 1:
            dfs = {r: df[df["state"].isin(REGIONS[r])] for r in regions}
        else:
            dfs[name] = df

    return dfs


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
        states = sum([REGIONS[region] for region in regions], [])
        df = df[df["state"].isin(states)]

    return df


# pylint: disable=no-member
# pylint: disable=unsubscriptable-object
# pylint: disable=unsupported-assignment-operation
@cache3.memoize()
def cache_table(project, path, y_var, x_var, recalc_table=None, recalc="off"):
    """Read in just a single table."""
    # Get config
    config = Config(project)

    # Get the table
    if recalc == "on":
        data = ReCalculatedData(
            config=Config(project)).build(
                path, recalc_table
        )
    else:
        data = read_file(path, project)

    # We want some consistent fields
    if "capacity" not in data.columns and "hybrid_capacity" in data.columns:
        data["capacity"] = data["hybrid_capacity"].copy()

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


@cache4.memoize()
def cache_timeseries(file, project, map_selection, chart_selection,
                     map_click=None, variable="rep_profiles_0"):
    """Read and store a timeseries data frame with site selections."""
    # Convert map and chart selections into site indices
    gids = point_filter(map_selection, chart_selection, map_click)

    # Read in data frame
    data = read_timeseries(file, project, gids, nsteps=None, variable=variable)

    return data


def calc_least_cost(paths, dst, composite_function="min",
                    composite_variable="total_lcoe"):
    """Build the single least cost table from a list of tables."""
    # Not including an overwrite option for now
    if not os.path.exists(dst):

        # Collect all data frames - biggest lift of all
        paths = [Path(path) for path in paths]
        dfs = []
        ncpu = min([len(paths), mp.cpu_count() - 1])
        with mp.pool.ThreadPool(ncpu) as pool:
            for data in tqdm(pool.imap(read_file, paths), total=ncpu):
                dfs.append(data)

        # Make one big data frame and save
        data = composite(
            dfs,
            composite_function=composite_function,
            composite_variable=composite_variable
        )
        data.to_csv(dst, index=False)

def find_capacity_column(supply_curve_df, cap_col_candidates=None):
    """Find the capacity column in a supply curve data frame.

    Identifies the capacity column in a supply curve dataframe from a list of
    candidate columns. If more than one of the candidate columns is found in
    the dataframe, only the first one that occurs will be returned.

    Parameters
    ----------
    supply_curve_df : pd.core.frame.DataFrame
        Supply curve data frame.
    cap_col_candidates : [list, None], optional
        Candidate capacity column names, by default None, which will result in
        using the candidate column names ["capacity", "capacity_mw",
        "capacity_mw_dc"].

    Returns
    -------
    str
        Name of capacity column.

    Raises
    ------
    ValueError
        Raises a ValueError if none of the candidate capacity columns are
        found in the input dataframe.
    """
    if cap_col_candidates is None:
        cap_col_candidates = ["capacity", "capacity_mw", "capacity_mw_dc"]

    cap_col = None
    for candidate in cap_col_candidates:
        if candidate in supply_curve_df.columns:
            cap_col = candidate
            break

    if cap_col is None:
        raise ValueError(
            "Could not find capacity column using candidate column names: "
            f"{cap_col_candidates} "
        )

    return cap_col


def get_sheet(file_name, sheet_name=None, header=0):
    """Read in/check available sheets from an excel spreadsheet file."""
    # Open file
    file = pd.ExcelFile(file_name)
    sheets = file.sheet_names

    # Run with no sheet_name for a list of available sheets
    if not sheet_name:
        print("No sheet specified, returning a list of available sheets.")
        return sheets
    if sheet_name not in sheets:
        raise ValueError(sheet_name + " not in file.")

    # Try to open sheet, print options if it fails
    try:
        table = file.parse(sheet_name=sheet_name, header=header)
    except XLRDError:
        print(sheet_name + " is not available. Available sheets:\n")
        for s in sheets:
            print("   " + s)

    return table


def read_file(file, project, nrows=None):
    """Read a CSV, Parquet, or HDF5 file. Only the meta read for HDF5.

    Parameters
    ----------
    file : str
        Path to a reV data frame. CSV, Parquet, and HDF5 formats accepted.
    nrows : int
        Number of rows to read in.

    Returns
    -------
    pd.core.frame.DataFrame
    """
    # Get extension
    ext = os.path.splitext(file)[-1]
    name = os.path.basename(file)

    # Check extension and read file
    if ext in (".parquet", ".pqt"):
        if nrows:
            pf = ParquetFile(file)
            rows = next(pf.iter_batches(batch_size=nrows))
            data = pa.Table.from_batches([rows]).to_pandas()
        else:
            data = pd.read_parquet(file)
    elif ext == ".csv":
        data = pd.read_csv(file, nrows=nrows, low_memory=False)
    elif ext == ".h5":
        with h5py.File(file, "r") as ds:
            if nrows:
                data = pd.DataFrame(ds["meta"][:nrows])
            else:
                data = pd.DataFrame(ds["meta"][:])
            decode(data)
    else:
        raise OSError(f"{file}'s extension not compatible at the moment.")

    # Assign a scenario name to the dataframe (useful for composite building)
    if "scenario" not in data:
        data["scenario"] = strip_rev_filename_endings(name)

    # Apply legacy mapping to column names
    config = Config(project)
    data = data.rename(columns=config.legacy_mapping)

    return data


def read_timeseries(file, project, gids=None, nsteps=None,
                    variable="rep_profiles_0"):
    # pylint: disable=no-member
    """Read in a time-series from an HDF5 file.

    Parameters
    ----------
    file : str
        Path to HDF5 file.
    project : str
        Name of project associated with this file.
    gids : list
        List of sc_point_gids to use to filter sites.
    nsteps : int
        Number of time-steps to read in.
    variable : str
        Name of the HDF5 data set to return.

    Returns
    -------
    pd.core.frame.DataFrame
        A apandas dataframe containing the time-series, datetime stamp,
        day, week, and month.
    """
    # Open file and pull out needed datasets (how to catch with context mgmt?)
    try:
        ds = h5py.File(file)
    except OSError:
        print(f"Could not read {file} with h5py.")
        logger.error("Could not read %s with h5py.", str(file))
        raise

    # Get meta, convert gids to index positions if needed
    meta = pd.DataFrame(ds["meta"][:])

    # Rename legacy columns
    config = Config(project)
    data = meta.rename(columns=config.legacy_mapping)

    # Set nsteps
    if nsteps is None:
        nsteps = ds["time_index"].shape[0]

    # Find site indices
    if gids is not None:
        meta = meta[meta["sc_point_gid"].isin(gids)]
    idx = list(meta.index)

    # If no time index found, raise error
    variables = list(ds)
    if not any("time_index" in var for var in variables):
        raise NotImplementedError("Cannot handle the time series formatting "
                                  f"in {file}.")

    # If dset is associated with a year time index, use that time index
    time_index = "time_index"
    if "-" in variable and "time_index" not in variables:
        year = int(variable.split("-")[-1])
        time_index = f"time_index-{year}"

    # Break down time entries
    time = [t.decode() for t in ds[time_index][:nsteps]]
    dtime = [dt.datetime.strptime(t, TIME_PATTERN) for t in time]
    minutes = [t.minute for t in dtime]
    hours = [t.hour for t in dtime]
    days = [t.timetuple().tm_yday for t in dtime]
    weeks = [t.isocalendar().week for t in dtime]
    months = [t.month for t in dtime]

    # Process target data set
    data = ds[variable][:nsteps, idx]
    data = data.mean(axis=1)

    # Get units
    units = None
    if "units" in ds[variable].attrs:
        units = ds[variable].attrs["units"]

    # Close dataset, how do we handle read errors with context management?
    ds.close()

    # Compile data frame
    data = pd.DataFrame({
        "time": time,
        "minute": minutes,
        "hour": hours,
        "daily": days,
        "weekly": weeks,
        "monthly": months,
        "units": units,
        "profile": data
    })

    return data


