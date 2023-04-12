# -*- coding: utf-8 -*-
"""Fixtures and setup for use across all tests."""
from pathlib import Path
import json

import pytest
from selenium.webdriver.chrome.options import Options
from click.testing import CliRunner
import pandas as pd
import geopandas as gpd

import reView.utils.config
from reView import TEST_DATA_DIR
from reView.utils.functions import load_project_configs


@pytest.fixture
def test_data_dir():
    """Return TEST_DATA_DIR as a `Path` object."""
    return Path(TEST_DATA_DIR)


@pytest.fixture
def test_bespoke_supply_curve():
    """Return bespoke-supply-curve.csv as a `Path` object."""
    bespoke_csv = Path(TEST_DATA_DIR).joinpath('bespoke-supply-curve.csv')

    return bespoke_csv


@pytest.fixture
def test_characterization_supply_curve():
    """Return characterization-supply-curve.csv as a `Path` object."""
    char_csv = Path(TEST_DATA_DIR).joinpath(
        'characterization-supply-curve.csv'
    )

    return char_csv


@pytest.fixture
def char_map():
    """Return characterization map"""

    char_map_path = Path(TEST_DATA_DIR).joinpath("characterization-map.json")
    with open(char_map_path, "r") as f:
        map_data = json.load(f)

    return map_data


@pytest.fixture
def test_cli_runner():
    """Return a click CliRunner for testing commands"""
    return CliRunner()


# pylint: disable=redefined-outer-name
@pytest.fixture
def test_config_dir(test_data_dir):
    """Return test config directory as a `Path` object."""
    return test_data_dir / "configs"


# pylint: disable=redefined-outer-name
@pytest.fixture(autouse=True)
def test_configs(test_config_dir):
    """Load test configs."""
    reView.utils.config.REVIEW_DATA_DIR = TEST_DATA_DIR
    old_configs = reView.utils.config.PROJECT_CONFIGS
    test_configs_ = load_project_configs(test_config_dir)
    reView.utils.config.PROJECT_CONFIGS = test_configs_

    yield

    reView.utils.config.PROJECT_CONFIGS = old_configs

@pytest.fixture
def background_gdf():
    """
    Return a geopandas geodataframe that is the dissolved boundaries from
    states.geojson. To be used as the "background" layer for
    utils.plots.map_geodataframe_column() tests.
    """

    state_boundaries_path = Path(TEST_DATA_DIR).joinpath(
        "plots", "states.geojson"
    )
    states_gdf = gpd.read_file(state_boundaries_path)
    states_dissolved = states_gdf.unary_union
    states_dissolved_gdf = gpd.GeoDataFrame(
        {"geometry": [states_dissolved]},
        crs=states_gdf.crs).explode(index_parts=False)

    return states_dissolved_gdf


@pytest.fixture
def states_gdf():
    """
    Return a geopandas geodataframe that is the states boundaries from
    states.geojson. To be used as the "boundary" layer for
    utils.plots.map_geodataframe_column() tests.
    """

    state_boundaries_path = Path(TEST_DATA_DIR).joinpath(
        "plots", "states.geojson"
    )
    states_gdf = gpd.read_file(state_boundaries_path)
    states_singlepart_gdf = states_gdf.explode(index_parts=True)

    return states_singlepart_gdf


@pytest.fixture
def supply_curve_gdf():
    """
    Return a geopandas geodataframe of points from a test supply curve
    consisting of results for just a few states.
    """

    supply_curve_path = Path(TEST_DATA_DIR).joinpath(
        "plots", "map-supply-curve.csv"
    )
    supply_curve_df = pd.read_csv(supply_curve_path)
    supply_curve_gdf = gpd.GeoDataFrame(
        supply_curve_df,
        geometry=gpd.points_from_xy(
            supply_curve_df['longitude'], supply_curve_df['latitude']
        ),
        crs="EPSG:4326"
    )

    return supply_curve_gdf


def pytest_setup_options():
    """Recommended setup based on https://dash.plotly.com/testing."""
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--headless")
    return options
