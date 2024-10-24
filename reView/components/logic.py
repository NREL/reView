# -*- coding: utf-8 -*-
"""Component logic functions."""
import json

import pandas as pd

from reView import Q_


def tab_styles(tab_choice, options):
    """Set correct tab styles for the chosen option.

    Parameters
    ----------
    tab_choice : str
        Name of the tab user selected.
    options : list
        List of option names user can select from.

    Returns
    -------
    list
        A list of styles for the tab options with the selected tab
        displayed and all others hidden.
    """
    styles = [{"display": "none"}] * len(options)
    idx = options.index(tab_choice)
    styles[idx] = {"width": "100%", "text-align": "center"}
    return styles


def format_capacity_title(
    map_capacity, map_selection=None, capacity_col_name="capacity_ac_mw"
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
            p.get("customdata", [None])[0] for p in map_selection["points"]     # Unintuitive
        ]
        df = df[df["sc_point_gid"].isin(gids)]

    total_capacity = Q_(df[capacity_col_name].sum(), "MW")                      # Q_ is also unintuitive
    total_capacity = f"{total_capacity.to_compact():~H.2f}"
    num_sites = f"{df.shape[0]:,}"

    return total_capacity, num_sites
