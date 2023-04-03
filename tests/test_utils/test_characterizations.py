# -*- coding: utf-8 -*-
"""Characterizations unit tests."""
import json

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from reView.utils.characterizations import (
    unpack_characterizations, validate_characterization_remapper
)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_unpack_characterizations_happy(
    test_characterization_supply_curve, test_data_dir
):
    """
    Happy path unit test for unpack_characterizations() function. Check that it
    produces expected output for provided input and characterization map.
    """

    in_df = pd.read_csv(test_characterization_supply_curve)

    char_map_path = test_data_dir.joinpath("characterization-map.json")
    with open(char_map_path, "r") as f:
        char_map = json.load(f)

    output_df = unpack_characterizations(in_df, char_map, cell_size_m=90)

    correct_results_src = test_data_dir.joinpath(
        'unpacked-characterization-supply-curve.csv'
    )
    correct_df = pd.read_csv(correct_results_src)

    assert_frame_equal(output_df, correct_df)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_unpack_characterizations_bad_method(
    test_characterization_supply_curve, test_data_dir
):
    """
    Test that unpack_characterizations() function correctly raises a ValueError
    when passed an invalid method. This is a proxy for testing that this
    function will catch various other invalidities in the input
    characterization map.
    """

    in_df = pd.read_csv(test_characterization_supply_curve)

    char_map_path = test_data_dir.joinpath("characterization-map.json")
    with open(char_map_path, "r") as f:
        char_map = json.load(f)

    char_map["nlcd_2019_90x90"]["method"] = "not-a-valid-method"
    with pytest.raises(ValueError):
        unpack_characterizations(in_df, char_map, cell_size_m=90)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_validate_characterization_remapper_happy(
    test_characterization_supply_curve, test_data_dir
):
    """
    Happy path test for validate_characterization_remapper(). Make sure it
    succeeds without raising errors for known test data and char map.
    """

    in_df = pd.read_csv(test_characterization_supply_curve)

    char_map_path = test_data_dir.joinpath("characterization-map.json")
    with open(char_map_path, "r") as f:
        char_map = json.load(f)

    validate_characterization_remapper(char_map, in_df)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_validate_characterization_remapper_key_error(
    test_characterization_supply_curve, test_data_dir
):
    """
    Test that validate_characterization_remapper() will raise a KeyError
    when passed a map column that does not exist in the input dataframe.
    """

    in_df = pd.read_csv(test_characterization_supply_curve)
    in_df.drop(columns=["fed_land_owner"], inplace=True)
    char_map_path = test_data_dir.joinpath("characterization-map.json")
    with open(char_map_path, "r") as f:
        char_map = json.load(f)

    with pytest.raises(KeyError):
        validate_characterization_remapper(char_map, in_df)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_validate_characterization_remapper_value_error(
    test_characterization_supply_curve, test_data_dir
):
    """
    Test that validate_characterization_remapper() will raise a ValueError
    when passed various invalid combinations of mappings.
    """

    in_df = pd.read_csv(test_characterization_supply_curve)

    char_map_path = test_data_dir.joinpath("characterization-map.json")
    with open(char_map_path, "r") as f:
        char_map_original = json.load(f)

    # not a valid method
    char_map_bad = char_map_original.copy()
    char_map_bad["fed_land_owner"]["method"] = "not-a-valid-method"
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # not a valid recast
    char_map_bad = char_map_original.copy()
    char_map_bad["fed_land_owner"]["recast"] = "not-a-valid-recast"
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # method = category but lkup is not a dict
    char_map_bad = char_map_original.copy()
    char_map_bad["fed_land_owner"]["lkup"] = "not-a-dictionary"
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # method = category but rename is not None
    char_map_bad = char_map_original.copy()
    char_map_bad["fed_land_owner"]["rename"] = "cannot-rename-this"
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # method = sum but lkup is not None
    char_map_bad = char_map_original.copy()
    char_map_bad["fed_land_owner"]["method"] = "sum"
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # method = sum, lkup is correct, but rename is not None or string
    char_map_bad["fed_land_owner"]["lkup"] = None
    char_map_bad["fed_land_owner"]["rename"] = {}
    with pytest.raises(ValueError):
        validate_characterization_remapper(char_map_bad, in_df)

    # method = None but any of the other values are not None
    char_map_bad = char_map_original.copy()
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


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
