# -*- coding: utf-8 -*-
"""Utilities for parsing user selection."""
import os
import logging

import pandas as pd

from reView.pages.rev.model import read_file
from reView.utils.config import Config
from reView.utils.functions import convert_to_title

logger = logging.getLogger(__name__)


def parse_selection(scenario_options):
    """_summary_

    Parameters
    ----------
    scenario_options : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if scenario_options is None:
        return {}
    selected_options = {}
    scenarios_div = scenario_options["props"]["children"]
    for option_div in scenarios_div:
        option_name_div = option_div["props"]["children"][0]
        title_div = option_name_div["props"]["children"][0]
        selection_name = title_div["props"]["children"]

        selection_div = option_div["props"]["children"][1]
        dropdown_div = selection_div["props"]["children"][0]
        selection_value = dropdown_div["props"]  # ["value"]
        selected_options[selection_name] = selection_value

    return selected_options


def all_files_from_selection(options, config):
    """_summary_

    Parameters
    ----------
    options : _type_
        _description_
    config : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    try:
        df = config.options.copy()
    except AttributeError:
        raise ValueError("Missing project options csv!") from None

    for key, val in options.items():
        df = df[df[key] == val["value"]]
    return df


def file_for_selections(selected_options, config):
    """_summary_

    Parameters
    ----------
    selected_options : _type_
        _description_
    config : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    row = all_files_from_selection(selected_options, config)
    logger.debug("row = %s", row)
    if "file" in row:
        logger.debug("file = %s", row["file"].values[0])
        return row["file"].values[0]

    name = row["name"].values[0]
    logger.debug("name = %s", name)
    file = config.files.get(f"{name}")
    logger.debug("file = %s", file)
    return file


def choose_scenario(scenario_options, config):
    """_summary_

    Parameters
    ----------
    scenario_options : _type_
        _description_
    config : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    selected_options = parse_selection(scenario_options)
    logger.debug("selected_options = %s", selected_options)
    if not selected_options:
        # TODO what if `files` is empty?
        # How to tell user no files found in dir?
        return list(config.files.values())[0]

    if "Scenario" in selected_options:
        return selected_options["Scenario"]["value"]

    return file_for_selections(selected_options, config)


def get_variable_options(
    project,
    scenario_a,
    scenario_b,
    b_div={"display": "none"}
):  # pylint: disable=dangerous-default-value
    """Retrieve appropriate variable list."""
    config = Config(project)
    variable_options = []
    if scenario_a and os.path.exists(scenario_a):
        columns = read_file(scenario_a, nrows=1).columns
        if "display" in b_div and b_div["display"] == "none":
            if scenario_b and os.path.exists(scenario_b):
                b_columns = read_file(scenario_b, nrows=1).columns
                columns = [c for c in columns if c in b_columns]
        titles = {col: convert_to_title(col) for col in columns}
        config_titles = {k: v for k, v in config.titles.items() if k in titles}
        titles.update(config_titles)
        if titles:
            for key, value in titles.items():
                variable_options.append({"label": value, "value": key})

    return variable_options
