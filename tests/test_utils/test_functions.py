# -*- coding: utf-8 -*-
"""Util function tests."""
import pytest
import pandas as pd

from reView.utils.functions import (
    convert_to_title,
    strip_rev_filename_endings,
    load_project_configs,
    adjust_cf_for_losses,
    common_numeric_columns,
)


@pytest.mark.parametrize(
    "given_input, expected_output",
    [
        (None, "None"),
        ("title", "Title"),
        ("A Correct Title", "A Correct Title"),
        ("a_title_with_underscores", "A Title With Underscores"),
        ("a title without underscores", "A Title Without Underscores"),
    ],
)
def test_convert_to_title(given_input, expected_output):
    """Test `convert_to_title` function."""
    assert convert_to_title(given_input) == expected_output


@pytest.mark.parametrize(
    "file",
    [
        "file_sc.csv",
        "file_agg.csv",
        "file_nrwal_00.csv",
        "file_nrwal_01.csv",
        "file_nrwal_10.csv",
        "file_supply-curve.csv",
        "file_supply-curve-aggregation.csv",
    ],
)
def test_strip_rev_filename_endings(file):
    """Test `strip_rev_filename_endings` function."""

    assert strip_rev_filename_endings(file) == "file"


def test_strip_rev_filename_endings_unknown_ending():
    """Test `strip_rev_filename_endings` with unknown file ending."""

    assert (
        strip_rev_filename_endings("file_generation.csv")
        == "file_generation.csv"
    )


def test_load_project_configs(test_config_dir):
    """Test `load_project_configs`."""

    configs = load_project_configs(test_config_dir)
    assert len(configs) >= 2
    assert "Test No Name" in configs
    assert "A test_project" in configs


def test_adjust_cf_for_losses():
    """Test `adjust_cf_for_losses` function."""
    assert adjust_cf_for_losses(0.1, 0.5, 0.25) == 0.1 * 0.5 / 0.75
    assert adjust_cf_for_losses(0.1, 0.5, 0) == 0.05


@pytest.mark.parametrize("bad_orig_losses", [-0.5, 1, 1.3])
def test_adjust_cf_for_losses_bad_input(bad_orig_losses):
    """Test `adjust_cf_for_losses` function for bad input."""
    with pytest.raises(ValueError) as excinfo:
        adjust_cf_for_losses(0.1, 0.5, bad_orig_losses)

    assert "Invalid input: `original_losses`" in str(excinfo.value)


def test_common_numeric_columns():
    """Test `common_numeric_columns` function. """

    df1 = pd.DataFrame({'a': [1], 'b': [345], 'c': ['Hello']})
    df2 = pd.DataFrame({'b': [1], 'c': ['Hello'], 'd': [2]})
    df3 = pd.DataFrame({'b': [1], 'c': ['Hello']}).iloc[[]]

    assert len(df3) == 0
    assert common_numeric_columns(df1) == ['a', 'b']
    assert common_numeric_columns(df1, df2) == ['b']
    assert common_numeric_columns(df1, df3) == ['b']
    assert common_numeric_columns(df1, df2, df3) == ['b']


