"""reView functions."""
# pylint: disable=broad-exception-caught
import ast
import json
import logging
import os
import re

from pathlib import Path

from pygeopkg.conversion.to_geopkg_geom import (
    point_to_gpkg_point,
    make_gpkg_geom_header
)
from pygeopkg.core.field import Field
from pygeopkg.core.geopkg import GeoPackage
from pygeopkg.core.srs import SRS
from pygeopkg.shared.constants import SHAPE
from pygeopkg.shared.enumeration import GeometryType, SQLFieldTypes

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
    """Return dictionary of posix path objects for reView package data.

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
    str : 
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
    """Attempt to replace values and recursively call `deep_replace`."""
    try:
        if value in replacement:
            dictionary[key] = replacement[value]
    except TypeError:  # `value` is not hashable, i.e. is a dict or mapping
        pass

    deep_replace(value, replacement)
