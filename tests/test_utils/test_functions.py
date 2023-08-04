# -*- coding: utf-8 -*-
"""Util function tests."""
import pytest
import pandas as pd

from reView.utils.functions import (
    convert_to_title,
    strip_rev_filename_endings,
    load_project_configs,
    data_paths,
    adjust_cf_for_losses,
    common_numeric_columns,
    deep_replace,
    shorten,
    as_float,
    safe_convert_percentage_to_decimal,
    find_capacity_column
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


def test_load_project_configs(config_dir_test):
    """Test `load_project_configs`."""

    configs = load_project_configs(config_dir_test)
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
    """Test `common_numeric_columns` function."""

    df1 = pd.DataFrame({'a': [1], 'b': [345], 'c': ['Hello']})
    df2 = pd.DataFrame({'b': [1], 'c': ['Hello'], 'd': [2]})
    df3 = pd.DataFrame({'b': [1], 'c': ['Hello']}).iloc[[]]

    assert len(df3) == 0
    assert common_numeric_columns(df1) == ['a', 'b']
    assert common_numeric_columns(df1, df2) == ['b']
    assert common_numeric_columns(df1, df3) == ['b']
    assert common_numeric_columns(df1, df2, df3) == ['b']


def test_deep_replace():
    """Test dictionary deep replacement."""
    in_dict = {
        'a': 'na',
        'b': {
            'a': 7,
            'b': 'na',
            'c': 'hello',
            'd': {
                5: 'na',
                6: 7,
                7: 8
            }
        },
        'c': 25,
        'd': 'goodbye'
    }
    mapping = {'na': None, 'hello': 'goodbye', 7: '7'}
    expected_output = {
        'a': None,
        'b': {
            'a': '7',
            'b': None,
            'c': 'goodbye',
            'd': {
                5: None,
                6: '7',
                7: 8
            }
        },
        'c': 25,
        'd': 'goodbye'
    }

    deep_replace(in_dict, mapping)
    assert in_dict == expected_output


def test_shorten():
    """Test the `shorten` function."""

    out = shorten('Hello, this is a long sentence', 10)

    assert len(out) == 10
    assert out == 'He...tence'

    out = shorten('Hello, this is a long sentence', 19)

    assert len(out) == 19
    assert out == 'Hello, this...tence'

    out = shorten('A word', 19)

    assert len(out) == 6
    assert out == 'A word'

    out = shorten('Hello, this is a long sentence', 19, inset=';;;')

    assert len(out) == 19
    assert out == 'Hello, this;;;tence'

    out = shorten('Hello, this is a long sentence', 19, chars_at_end=10)

    assert len(out) == 19
    assert out == 'Hello,...g sentence'


def test_data_paths():
    """test `data_paths` function."""

    paths = data_paths()

    assert 'reeds' in paths

    for name, path in paths.items():
        assert name == path.name
        assert path.exists()


def test_as_float():
    """Test `as_float` function."""

    assert isinstance(as_float("2000"), float)
    assert as_float("2000") == 2000
    assert as_float("2,000") == 2000
    assert as_float("2,000.54") == 2000.54
    assert as_float("$2,000.54") == 2000.54
    assert as_float("2,0.54%") == 20.54


def test_safe_convert_percentage_to_decimal():
    """Test safe percentage converter. """

    assert safe_convert_percentage_to_decimal(96) == 0.96
    assert safe_convert_percentage_to_decimal(96.54) == 0.9654
    assert safe_convert_percentage_to_decimal(1) == 1
    assert safe_convert_percentage_to_decimal(0.5) == 0.5


def test_find_capacity_column_defaults(
    map_supply_curve_solar, map_supply_curve_wind
):
    """
    Tests that find_capacity_column() returns the expected capacity column for
    a real solar and a real wind supply curve, based on the default list of
    candidate column names
    """
    solar_df = pd.read_csv(map_supply_curve_solar)
    wind_df = pd.read_csv(map_supply_curve_wind)

    assert find_capacity_column(solar_df) == "capacity_mw_dc"
    assert find_capacity_column(wind_df) == "capacity_mw"


def test_find_capacity_column_solar_ac_first(map_supply_curve_solar):
    """
    Tests that find_capacity_column() returns the ac capacity column for
    a real solar supply curve when the candidate list starts with
    capacity_mw_ac.
    """
    solar_df = pd.read_csv(map_supply_curve_solar)

    candidates = ["capacity_mw_ac", "capacity_mw_dc"]
    cap_col = find_capacity_column(solar_df, cap_col_candidates=candidates)
    assert cap_col == "capacity_mw_ac"


def test_find_capacity_column_none_found(map_supply_curve_solar):
    """
    Tests that find_capacity_column() raises a ValueError when none of the
    candidate columns are found in the input dataframe.
    """

    solar_df = pd.read_csv(map_supply_curve_solar)

    candidates = ["capacity", "capacity_mw"]
    with pytest.raises(ValueError):
        find_capacity_column(solar_df, cap_col_candidates=candidates)
