"""reView functions."""
import json
import os
import re
import logging
from pathlib import Path

import numpy as np

from reView import REVIEW_CONFIG_DIR

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
    """Calcaulate new cf based on old and new loss assumptions.

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


def common_numeric_columns(*dataframes):
    """Find all common numeric columns in input DataFrames.

    Parameters
    ----------
    *dataframes
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
          for df in dataframes]
    )
    return sorted(cols)
