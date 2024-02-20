"""reView functions."""
# pylint: disable=broad-exception-caught
import ast
import datetime as dt
import json
import logging
import os
import re

from pathlib import Path

import h5py
import pandas as pd
import pyarrow as pa

from pyarrow.parquet import ParquetFile
from pygeopkg.conversion.to_geopkg_geom import (
    point_to_gpkg_point,
    make_gpkg_geom_header
)
from pygeopkg.core.field import Field
from pygeopkg.core.geopkg import GeoPackage
from pygeopkg.core.srs import SRS
from pygeopkg.shared.constants import SHAPE
from pygeopkg.shared.enumeration import GeometryType, SQLFieldTypes
from xlrd import XLRDError

import dash
import numpy as np
import pyproj

from reView import REVIEW_CONFIG_DIR, REVIEW_DATA_DIR
from reView.paths import Paths

logger = logging.getLogger(__name__)


TIME_PATTERN = "%Y-%m-%d %H:%M:%S+00:00"


def adjust_cf_for_losses(mean_cf, new_losses, original_losses):
    """Calculate new cf based on old and new loss assumptions.

    Parameters
    ----------
    mean_cf : float
        Input capacity factor.
    new_losses : float
        New loss multiplier value (e.g. 20% losses = 0.2).
    original_losses : float
        Original loss multiplier value (e.g. 15% losses = 0.15).
        Must be in the range [0, 1)!

    Returns
    -------
    float
        Capacity factor adjusted for new losses.
    """
    if not 0 <= original_losses < 1:
        msg = (
            f"Invalid input: `original_losses`={original_losses}. "
            f"Must be in the range [0, 1)!"
        )
        logger.error(msg)
        raise ValueError(msg)

    gross_cf = mean_cf / (1 - original_losses)
    mean_cf = gross_cf - (gross_cf * new_losses)

    return mean_cf


def as_float(value):
    """Convert a string representation of float to float.

    In particular, the string representation can have commas, dollar
    signs, and percent signs. All of these will be stripped, and the
    final result will be converted to a float.

    Parameters
    ----------
    value : _type_
        Input string that may contain commas, dollar signs, and percent
        signs. Should also contain a float value, which will be returned
        as a float object.

    Returns
    -------
    float
        Input string value represented as a float.
    """
    if isinstance(value, str):
        value = value.replace(",", "").replace("$", "").replace("%", "")
        value = float(value)
    return value


def callback_trigger():
    """Get the callback trigger, if it exists.

    Returns
    -------
    str
        String representation of callback trigger, or "Unknown" if
        context not found.
    """

    try:
        trigger = dash.callback_context.triggered[0]
        trigger = trigger["prop_id"]
    except dash.exceptions.MissingCallbackContextException:
        trigger = "Unknown"

    return trigger


def capacity_factor_from_lcoe(capacity, mean_lcoe, calc_values):
    """Calculate teh capacity factor given the lcoe.

    Parameters
    ----------
    capacity : `pd.core.series.Series` | `np.ndarray`
        A series of capacity values, in MW.
    mean_lcoe : `pd.core.series.Series` | `np.ndarray`
        A series of lcoe values, in $/MW.
    calc_values : dict
        A dictionary with entries for capex ($/kW), opex ($/kW),
        and fcr (decimal - 4.9% should be input as 0.049).

    Returns
    -------
    float
        Capacity factor calculated from lcoe.
    """
    capacity_kw = capacity * 1000
    capex = calc_values["capex"] * capacity_kw
    opex = calc_values["opex"] * capacity_kw
    fcr = calc_values["fcr"]
    return ((fcr * capex) + opex) / (mean_lcoe * capacity * 8760)


def common_numeric_columns(*dfs):
    """Find all common numeric columns in input DataFrames.

    Parameters
    ----------
    *dfs
        One or more pandas DataFrame objects to compare.
        The common numeric columns from these inputs will be returned.

    Returns
    -------
    list
        A sorted list of the common numeric columns among the input
        DataFrames.
    """
    cols = set.intersection(
        *[
            set(df.select_dtypes(include=np.number).columns.values)
            for df in dfs
        ]
    )
    return sorted(cols)


def convert_to_title(col_name):
    """Turn a column name into a title.

    This function replaces underscores with spaces and capitalizes the
    resulting words.

    Parameters
    ----------
    col_name : str
        Input column name.

    Returns
    -------
    str | None
        Output column name formatted into title. Returns the string
        "None" if `col_name` input is None.

    Examples
    --------
    Simple use cases:

    >>> convert_to_title(None)
    'None'
    >>> convert_to_title('title')
    'Title'
    >>> convert_to_title('a_title_with_underscores')
    'A Title With Underscores'
    """
    if col_name is None:
        return "None"
    return col_name.replace("_", " ").title()


def data_paths():
    """Dictionary of posix path objects for reView package data.

    Returns
    -------
    dict
        A dictionary mapping data folder names (keys) to data folder
        paths.
    """
    return {
        folder.name.lower(): folder
        for folder in Path(REVIEW_DATA_DIR).iterdir()
        if not folder.is_file()
    }


def decode(df):
    """Decode the columns of a meta data object from a reV output."""
    def decode_single(x):
        """Try to decode a single value, pass if fail."""
        try:
            x = x.decode()
        except UnicodeDecodeError:
            x = "indecipherable"
        return x

    for c in df.columns:
        x = df[c].iloc[0]
        if isinstance(x, bytes):
            try:
                df[c] = df[c].apply(decode_single)
            except Exception:
                df[c] = None
                print(f"Column {c} could not be decoded.")
        elif isinstance(x, str):
            try:
                if isinstance(ast.literal_eval(x), bytes):
                    try:
                        df[c] = df[c].apply(
                            lambda x: ast.literal_eval(x).decode()
                            )
                    except Exception:
                        df[c] = None
                        print(f"Column {c} could not be decoded.")
            except Exception:
                pass


def deep_replace(dictionary, replacement):
    """Perform a deep replacement in the dictionary using the mapping.

    This function performs inplace replacement of values in the input
    dictionary. The function is recursive, so the mapping will be
    applied to all nested dictionaries.

    Parameters
    ----------
    dictionary : dict
        Input dictionary with values that need to be replaced.
    replacement : dict
        Mapping where keys represent the target values to be replaced,
        and the corresponding values are the replacements.

    Examples
    --------
    >>> a = {'a': 'na', 'b': {'c': 'na', 'd': 5}}
    >>> mapping = {'na': None}
    >>> deep_replace(a, mapping)
    >>> a
    {'a': None, 'b': {'c': None, 'd': 5}}
    """
    try:
        for key, value in dictionary.items():
            __replace_value(dictionary, replacement, key, value)
    except AttributeError:  # `dictionary`` does not have `.items()` method
        return


def get_project_defaults():
    """Get the default project for each page from a file (easier to change)."""
    fpath = Paths.home.joinpath("reView/default_project")
    with open(fpath, "r", encoding="utf-8") as file:
        defaults = json.load(file)
    return defaults


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


def is_int(val):
    """Check if an input value is an integer."""
    try:
        int(val)
        return True
    except ValueError:
        return False


def lcoe(capacity, mean_cf, re_calcs):
    """Calculate Levelized Cost of Energy (LCOE).

    Parameters
    ----------
    capacity : `pd.core.series.Series` | `np.ndarray`
        A series of capacity values, in MW.
    mean_cf : `pd.core.series.Series` | `np.ndarray`
        A series of capacity factor values, given as fractional ratio.
    re_calcs : dict
        A dictionary with new entries for capex ($/kW), opex ($/kW),
        and fcr (decimal - 4.9% should be input as 0.049).

    Returns
    -------
    np.ndarray
        A series of LCOE values.
    """
    capacity_kw = capacity * 1000
    capex = re_calcs["capex"] * capacity_kw
    opex = re_calcs["opex"] * capacity_kw
    fcr = re_calcs["fcr"]
    lcoe_ = ((fcr * capex) + opex) / (capacity * mean_cf * 8760)
    return lcoe_


def lcot(capacity, trans_cap_cost, mean_cf, re_calcs):
    """Calculate Levelized Cost of Transportation (LCOT).

    Parameters
    ----------
    capacity : `pd.core.series.Series` | `np.ndarray`
        A series of capacity values, in MW.
    trans_cap_cost : `pd.core.series.Series` | `np.ndarray`
        A series of transmission capital cost values ($/MW).
    mean_cf : `pd.core.series.Series` | `np.ndarray`
        A series of capacity factor values, given as fractional ratio.
    re_calcs : dict
        A dictionary with new entries for capex ($/kW), opex ($/kW),
        and fcr (decimal - 4.9% should be input as 0.049).

    Returns
    -------
    np.ndarray
        A series of LCOT values.
    """
    fcr = re_calcs["fcr"]
    capex = trans_cap_cost * capacity
    lcot_ = (capex * fcr) / (capacity * mean_cf * 8760)
    return lcot_


def load_project_configs(config_dir=REVIEW_CONFIG_DIR):
    """Load projects from configs.

    Parameters
    ----------
    config_dir : str, optional
        Path to directory containing project config files. By default
        `REVIEW_CONFIG_DIR`.

    Returns
    -------
    project_configs : dict
        Dictionary where keys represent the project names and values are
        the corresponding project configs.
    """
    project_configs = {}
    files = Path(config_dir).glob("*.json")
    for file in sorted(files, key=lambda t: os.stat(t).st_mtime):
        with open(file, "r") as file_handle:
            config = json.load(file_handle)
            file_name_no_extension = ".".join(file.name.split(".")[:-1])
            project_name = config.get(
                "project_name", convert_to_title(file_name_no_extension)
            )
            project_configs[project_name] = config
    return project_configs


def read_file(file, nrows=None):
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

    return data


def read_timeseries(file, gids=None, nsteps=None):
    # pylint: disable=no-member
    """Read in a time-series from an HDF5 file.

    Parameters
    ----------
    file : str
        Path to HDF5 file.
    gids : list
        List of sc_point_gids to use to filter sites.
    nsteps : int
        Number of time-steps to read in.

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

    # Set nsteps
    if nsteps is None:
        nsteps = ds["time_index"].shape[0]

    # Find site indices
    if gids is not None:
        meta = meta[meta["sc_point_gid"].isin(gids)]
    idx = list(meta.index)

    # Get capacity, time index, format
    capacity = meta["capacity"].values

    # If it has any "rep_profiles_" datasets it rep-profiles
    if "bespoke" not in str(file):
        # Break down time entries
        time = [t.decode() for t in ds["time_index"][:nsteps]]
        dtime = [dt.datetime.strptime(t, TIME_PATTERN) for t in time]
        minutes = [t.minute for t in dtime]
        hours = [t.hour for t in dtime]
        days = [t.timetuple().tm_yday for t in dtime]
        weeks = [t.isocalendar().week for t in dtime]
        months = [t.month for t in dtime]

        # Process generation data
        cf = ds["rep_profiles_0"][:nsteps, idx]
        gen = cf * capacity
        cf = cf.mean(axis=1)
        gen = gen.sum(axis=1)

    # Otherwise, it's bespoke and has each year
    else:
        # Get all capacity factor keys
        cf_keys = [key for key in ds.keys() if "cf_profile-" in key]
        time_keys = [key for key in ds.keys() if "time_index-" in key]
        scale = ds[cf_keys[0]].attrs["scale_factor"]

        # Build complete time-series at each site
        all_cfs = []
        all_time = []
        for i, cf_key in enumerate(cf_keys):
            time_key = time_keys[i]
            cf = ds[cf_key][:nsteps, idx]
            time = ds[time_key][:nsteps]
            all_cfs.append(cf)
            all_time.append(time)
        site_cfs = np.concatenate(all_cfs)
        time = np.concatenate(all_time)
        site_gen = site_cfs * capacity

        # Build single long-term average timeseries for all sites
        cf = np.mean(site_cfs, axis=1) / scale
        gen = site_gen.sum(axis=1)

        # This will only take the average across the year
        time = [t.decode() for t in time]
        dtime = [dt.datetime.strptime(t, TIME_PATTERN) for t in time]
        days = [t.timetuple().tm_yday for t in dtime]
        weeks = [t.isocalendar().week for t in dtime]
        months = [t.month for t in dtime]
        hours = [t.hour for t in dtime]
        minutes = [t.minute for t in dtime]

    ds.close()

    data = pd.DataFrame({
        "time": time,
        "minute": minutes,
        "hour": hours,
        "daily": days,
        "weekly": weeks,
        "monthly": months,
        "capacity factor": cf,
        "generation": gen
    })

    return data


def safe_convert_percentage_to_decimal(value):
    """Convert a percentage value to a decimal if it is > 1%.

    *IMPORTANT*
    This function assumes any input less than 1 is already a decimal
    and does not alter the input.

    Parameters
    ----------
    value : float
        Input percentage value. Must be greater than 1% in order for
        the conversion to be applied.

    Returns
    -------
    float
        Input percentage value as decimal.
    """
    if value > 1:
        value = value / 100
    return value


def shorten(string, new_length, inset="...", chars_at_end=5):
    """Shorten a long string.

    This function truncates the middle of the string and replaces it
    with `inset` such that the output string is (at most) of length
    `new_length`.

    Parameters
    ----------
    string : string
        Input string. Can be shorter than `new_length`, in which case
        the string is returned unaltered.
    new_length : int
        Desired length of the return string.
    inset : str, optional
        Inset to use in the middle of the string to indicate that it was
        truncated. By default, `"..."`.
    chars_at_end : int, optional
        Number of characters to leave at end of truncated string.
        By default, `5`.

    Returns
    -------
    str
        Shortened string.
    """

    if len(string) <= new_length:
        return string

    len_first_part = min(len(string), new_length)
    len_first_part = max(0, len_first_part - chars_at_end - len(inset))
    return f"{string[:len_first_part]}{inset}{string[-chars_at_end:]}"


def strip_rev_filename_endings(filename):
    """Strip file endings from reV output files.

    The following file endings are removed:
        - "_sc.csv"
        - "_agg.csv"
        - "_nrwal*.csv"
        - "_supply-curve.csv"
        - "_supply-curve-aggregation.csv"

    If a file ends in ".csv" but does not match one of these patterns,
    the file name will remain unchanged.

    Parameters
    ----------
    filename : str
        Input filename as string.

    Returns
    -------
    str
        Filename without the file endings listed above.

    Examples
    --------
    Simple use cases:

    >>> strip_rev_filename_endings('input_file_nrwal_01.csv')
    'input_file'
    >>> strip_rev_filename_endings('name_supply-curve-aggregation.csv')
    'name'
    """

    patterns = [
        r"_sc\.csv",
        r"_agg\.csv",
        r"_nrwal.*\.csv",
        r"_supply-curve\.csv",
        r"_supply-curve-aggregation\.csv",
        r"_supply_curve\.csv",
        r"_supply_curve_aggregation\.csv",
        r"_sc\.parquet",
        r"_agg\.parquet",
        r"_nrwal.*\.parquet",
        r"_supply-curve\.parquet",
        r"_supply-curve-aggregation\.parquet",
        r"_supply_curve\.parquet",
        r"_supply_curve_aggregation\.parquet",
        r"\.h5"
    ]
    full_pattern = "|".join(patterns)
    return re.sub(full_pattern, "", filename)


def to_geo(df, dst, layer):
    # pylint: disable=too-many-branches
    """Convert pandas data frame to geodataframe.

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        A pandas datasframe with "latitude" and "longitude" coordinate fields.
    dst : str | pathlib.PosixPath
        Destination path for output.
    layer : str
        Layer name.
    """
    # Initialize file
    gpkg = GeoPackage.create(dst, flavor="EPSG")

    # Create spatial references
    wkt = pyproj.CRS("epsg:4326").to_wkt()
    srs = SRS("WGS_1984", "EPSG", 4326, wkt)

    # The index field breaks it
    if "index" in df:
        del df["index"]

    # Remove or rename columns
    replacements = {
        "-": "_",
        " ": "_",
        "/": "_",
        "$": "usd",
        "?": "",
        "(": "",
        ")": "",
        "%": "pct",
        "&": "and"
    }
    for col in df.columns:
        # Remove columns that start with numbers
        if is_int(col[0]):
            del df[col]
            print(col)

        # This happens when you save the index
        if "Unnamed:" in col:
            del df[col]
        else:
            # Remove unnacceptable characters
            ncol = col
            for char, repl in replacements.items():
                ncol = ncol.replace(char, repl)

            # Lower case just because
            ncol = ncol.lower()

            # Columns also can't start with an integer
            # parts = ncol.split("_")
            # for part in parts:
            #     if is_int(part):
            #         npart1 = "_".join(ncol.split("_")[1:])
            #         npart2 = ncol.split("_")[0]
            #         ncol = "_".join([npart1, npart2])

            # Rename column
            if col != ncol:
                df = df.rename({col: ncol}, axis=1)

    # Create fields and set types
    fields = []
    for col, values in df.items():
        dtype = str(values.dtype)
        if "int" in dtype:
            ftype = SQLFieldTypes.integer
        elif "float" in dtype:
            ftype = SQLFieldTypes.float
        elif dtype == "object":
            ftype = SQLFieldTypes.text
        elif dtype == "bool":
            ftype = SQLFieldTypes.boolean
        else:
            raise TypeError("Could not determine data type of values for "
                            f"{col} column.")
        fields.append(Field(col, ftype))

    # Create feature class
    layer = layer.replace("-", "_")
    features = gpkg.create_feature_class(
        name=layer,
        srs=srs,
        fields=fields,
        shape_type=GeometryType.point
    )

    # Build data rows
    header = make_gpkg_geom_header(features.srs.srs_id)
    field_names = list(df.columns)
    field_names.insert(0, SHAPE)
    rows = []
    for _, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        wkb = point_to_gpkg_point(header, lon, lat)
        values = list(row.values)
        values.insert(0, wkb)
        rows.append(values)

    # Finally insert rows
    features.insert_rows(field_names, rows)
    del features
    del gpkg


def to_sarray(df):
    """Create a structured array for storing in HDF5 files."""
    # For a single column
    def make_col_type(col, types):

        coltype = types[col]
        column = df.loc[:, col]

        try:
            if 'numpy.object_' in str(coltype.type):
                maxlens = column.dropna().str.len()
                if maxlens.any():
                    maxlen = maxlens.max().astype(int)
                    coltype = f'S{maxlen}'
                else:
                    coltype = 'f2'
            return column.name, coltype
        except:
            print(column.name, coltype, coltype.type, type(column))
            raise

    # All values and types
    v = df.values
    types = df.dtypes
    struct_types = [make_col_type(col, types) for col in df.columns]
    dtypes = np.dtype(struct_types)

    # The target empty array
    array = np.zeros(v.shape[0], dtypes)

    # For each type fill in the empty array
    for (i, k) in enumerate(array.dtype.names):
        try:
            if dtypes[i].str.startswith('|S'):
                array[k] = df[k].str.encode('utf-8').astype('S')
            else:
                array[k] = v[:, i]
        except Exception as e:
            raise e

    return array, dtypes


def __replace_value(dictionary, replacement, key, value):
    """Attempts replacement and recursively calls `deep_replace`."""
    try:
        if value in replacement:
            dictionary[key] = replacement[value]
    except TypeError:  # `value` is not hashable, i.e. is a dict or mapping
        pass

    deep_replace(value, replacement)


def find_capacity_column(supply_curve_df, cap_col_candidates=None):
    """
    Identifies the capacity column in a supply curve dataframe from a list of
    candidate columns. If more than one of the candidate columns is found in
    the dataframe, only the first one that occurs will be returned.

    Parameters
    ----------
    supply_curve_df : pandas.DataFrame
        Supply curve data frame
    cap_col_candidates : [list, None], optional
        Candidate capacity column names, by default None, which will result in
        using the candidate column names ["capacity", "capacity_mw",
        "capacity_mw_dc"].

    Returns
    -------
    str
        Name of capacity column

    Raises
    ------
    ValueError
        Raises a ValueError if none of the candidate capacity columns are
        found in the input dataframe.
    """
    if cap_col_candidates is None:
        cap_col_candidates = [
            "capacity", "capacity_mw", "capacity_mw_dc"
        ]

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
