# -*- coding: utf-8 -*-
"""Characterizations unit tests."""
import json

import numpy as np
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from reView.utils.characterizations import (
    unpack_characterizations, validate_characterization_remapper,
    recast_categories
)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_unpack_characterizations_happy(
    characterization_supply_curve, data_dir_test, char_map
):
    """
    Happy path unit test for unpack_characterizations() function. Check that it
    produces expected output for provided input and characterization map.
    """

    in_df = pd.read_csv(characterization_supply_curve)
    output_df = unpack_characterizations(in_df, char_map, cell_size_m=90)

    correct_results_src = data_dir_test.joinpath(
        'unpacked-characterization-supply-curve.csv'
    )
    correct_df = pd.read_csv(correct_results_src)

    assert_frame_equal(output_df, correct_df)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_unpack_characterizations_bad_method(
    characterization_supply_curve, char_map
):
    """
    Test that unpack_characterizations() function correctly raises a ValueError
    when passed an invalid method. This is a proxy for testing that this
    function will catch various other invalidities in the input
    characterization map, which are tested more thoroughly in other unit tests.
    """

    in_df = pd.read_csv(characterization_supply_curve)

    char_map["nlcd_2019_90x90"]["method"] = "not-a-valid-method"
    with pytest.raises(ValueError):
        unpack_characterizations(in_df, char_map, cell_size_m=90)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_validate_characterization_remapper_happy(
    characterization_supply_curve, char_map
):
    """
    Happy path test for validate_characterization_remapper(). Make sure it
    succeeds without raising errors for known test data and char map.
    """

    in_df = pd.read_csv(characterization_supply_curve)

    validate_characterization_remapper(char_map, in_df)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_validate_characterization_remapper_key_error(
    characterization_supply_curve, char_map
):
    """
    Test that validate_characterization_remapper() will raise a KeyError
    when passed a map column that does not exist in the input dataframe.
    """

    in_df = pd.read_csv(characterization_supply_curve)
    in_df.drop(columns=["fed_land_owner"], inplace=True)

    with pytest.raises(KeyError):
        validate_characterization_remapper(char_map, in_df)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_validate_characterization_remapper_value_error(
    characterization_supply_curve, char_map
):
    """
    Test that validate_characterization_remapper() will raise a ValueError
    when passed various invalid combinations of mappings.
    """

    in_df = pd.read_csv(characterization_supply_curve)

    # not a valid method
    char_map_bad = char_map.copy()
    char_map_bad["fed_land_owner"]["method"] = "not-a-valid-method"
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # not a valid recast
    char_map_bad = char_map.copy()
    char_map_bad["fed_land_owner"]["recast"] = "not-a-valid-recast"
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # method = category but lkup is not a dict
    char_map_bad = char_map.copy()
    char_map_bad["fed_land_owner"]["lkup"] = "not-a-dictionary"
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # method = category but rename is not None
    char_map_bad = char_map.copy()
    char_map_bad["fed_land_owner"]["rename"] = "cannot-rename-this"
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # method = sum but lkup is not None
    char_map_bad = char_map.copy()
    char_map_bad["fed_land_owner"]["method"] = "sum"
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # method = sum, lkup is correct, but rename is not None or string
    char_map_bad["fed_land_owner"]["lkup"] = None
    char_map_bad["fed_land_owner"]["rename"] = {}
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # method = None but any of the other values are not None
    char_map_bad = char_map.copy()
    char_map_bad["fed_land_owner"]["method"] = None
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)
    char_map_bad["fed_land_owner"]["lkup"] = None
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)
    char_map_bad["fed_land_owner"]["recast"] = None
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)
    char_map_bad["fed_land_owner"]["rename"] = None
    # this one should pass finally
    validate_characterization_remapper(char_map_bad, in_df)


def test_recast_categories_pass_through(char_map):
    """
    Test that recast_categories() unpacks data correctly (as pass through, no
    area recast).
    """

    col = "fed_land_owner"
    cell_size_sq_km = None
    lkup = char_map[col]["lkup"]

    mock_df = pd.DataFrame()
    mock_df["sc_gid"] = np.arange(1, 6)
    mock_char = {"255.0": 10, "4.0": 1, "6.0": 2}
    mock_df[col] = json.dumps(mock_char)

    mock_df = recast_categories(
        mock_df, col, lkup, cell_size_sq_km
    )
    for k, v in mock_char.items():
        assert np.all(mock_df[lkup[k]] == v)


def test_recast_categories_recast_to_area(char_map):
    """
    Test that recast_categories() unpacks data correct when recasting to area
    values.
    """

    col = "fed_land_owner"
    cell_size_sq_km = 90
    lkup = char_map[col]["lkup"]

    mock_df = pd.DataFrame()
    mock_df["sc_gid"] = np.arange(1, 6)
    mock_char = {"255.0": 10, "4.0": 1, "6.0": 2}
    mock_df[col] = json.dumps(mock_char)

    mock_df = recast_categories(
        mock_df, col, lkup, cell_size_sq_km
    )
    for k, v in mock_char.items():
        assert np.all(
            mock_df[f"{lkup[k]}_area_sq_km"] == v * cell_size_sq_km
        )


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
