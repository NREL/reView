"""reView functions."""
import json
import os
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import dash

from reView import REVIEW_CONFIG_DIR, REVIEW_DATA_DIR

logger = logging.getLogger(__name__)


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
    # pylint: disable=anomalous-backslash-in-string
    patterns = [
        "_sc\.csv",
        "_agg\.csv",
        "_nrwal.*\.csv",
        "_supply-curve\.csv",
        "_supply-curve-aggregation\.csv",
    ]
    full_pattern = "|".join(patterns)
    return re.sub(full_pattern, "", filename)


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
    }


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
        *[set(df.select_dtypes(include=np.number).columns.values)
          for df in dfs]
    )
    return sorted(cols)


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


def __replace_value(dictionary, replacement, key, value):
    """Attempts replacement and recursively calls `deep_replace`."""
    try:
        if value in replacement:
            dictionary[key] = replacement[value]
    except TypeError:  # `value` is not hashable, i.e. is a dict or mapping
        pass

    deep_replace(value, replacement)


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
        trigger = trigger["prop_id"].split(".")[0]
    except dash.exceptions.MissingCallbackContextException:
        trigger = "Unknown"

    return trigger


def format_capacity_title(
    map_capacity,
    map_selection=None,
    capacity_col_name="print_capacity"
):
    """Calculate total remaining capacity after all filters are applied.

    Parameters
    ----------
    map_capacity : str
        Serialized dictionary containing data. This input will be loaded
        as a pandas DataFrame and used to calculate capacity.
    map_selection : dict, optional
        Dictionary with a "points" key containing a list of the selected
        points, which have a `customdata` with gid values attached. By
        default, `None`.
    capacity_col_name : str, optional
        Name of column containing capacity values. By default,
        "print_capacity".

    Returns
    -------
    str
        Total capacity, formatted as a string.
    str
        Number of selected sites, formatted as a string.
    """

    if not map_capacity:
        return "--", "--"

    df = pd.DataFrame(json.loads(map_capacity))
    if df.empty:
        return "--", "--"

    if map_selection:
        gids = [
            p.get("customdata", [None])[0]
            for p in map_selection["points"]
        ]
        df = df[df["sc_point_gid"].isin(gids)]

    total_capacity = df[capacity_col_name].sum()
    if total_capacity >= 1_000_000:
        capacity = f"{total_capacity / 1_000_000:.4f} TW"
    else:
        capacity = f"{total_capacity / 1_000:.4f} GW"

    num_sites = f"{df.shape[0]:,}"
    return capacity, num_sites
