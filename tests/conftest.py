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

from tests import helper


@pytest.fixture
def data_dir_test():
    """Return TEST_DATA_DIR as a `Path` object."""
    return Path(TEST_DATA_DIR)


@pytest.fixture
def bespoke_supply_curve():
    """Return bespoke-supply-curve.csv as a `Path` object."""
    bespoke_csv = Path(TEST_DATA_DIR).joinpath('bespoke-supply-curve.csv')

    return bespoke_csv


@pytest.fixture
def characterization_supply_curve():
    """Return characterization-supply-curve.csv as a `Path` object."""
    char_csv = Path(TEST_DATA_DIR).joinpath(
        'characterization-supply-curve.csv'
    )

    return char_csv


@pytest.fixture
def map_supply_curve_solar():
    """Return plots/map-supply-curve-solar.csv as a `Path` object."""
    csv_path = Path(TEST_DATA_DIR).joinpath(
        "plots", "map-supply-curve-solar.csv"
    )

    return csv_path


@pytest.fixture
def map_supply_curve_wind():
    """Return plots/map-supply-curve-wind.csv as a `Path` object."""
    csv_path = Path(TEST_DATA_DIR).joinpath(
        "plots", "map-supply-curve-wind.csv"
    )

    return csv_path


@pytest.fixture
def map_supply_curve_wind_zeros():
    """Return plots/map-supply-curve-wind_zeros.csv as a `Path` object."""
    csv_path = Path(TEST_DATA_DIR).joinpath(
        "plots", "map-supply-curve-wind_zeros.csv"
    )

    return csv_path

@pytest.fixture
def char_map():
    """Return characterization map"""

    char_map_path = Path(TEST_DATA_DIR).joinpath("characterization-map.json")
    with open(char_map_path, "r") as f:
        map_data = json.load(f)

    return map_data


@pytest.fixture
def cli_runner():
    """Return a click CliRunner for testing commands"""
    return CliRunner()


# pylint: disable=redefined-outer-name
@pytest.fixture
def config_dir_test(data_dir_test):
    """Return test config directory as a `Path` object."""
    return data_dir_test / "configs"


@pytest.fixture(autouse=True)
def configs_test(config_dir_test):
    # pylint: disable=redefined-outer-name
    """Load test configs."""
    reView.utils.config.REVIEW_DATA_DIR = TEST_DATA_DIR
    old_configs = reView.utils.config.PROJECT_CONFIGS
    test_configs_ = load_project_configs(config_dir_test)
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
def county_background_gdf():
    """
    Return a geopandas geodataframe that is the dissolved boundaries from
    counties.geojson. To be used as the "background" layer for
    utils.plots.map_geodataframe_column() tests.
    """

    county_boundaries_path = Path(TEST_DATA_DIR).joinpath(
        "plots", "counties.geojson"
    )
    counties_gdf = gpd.read_file(county_boundaries_path)
    counties_dissolved = counties_gdf.unary_union
    counties_dissolved_gdf = gpd.GeoDataFrame(
        {"geometry": [counties_dissolved]},
        crs=counties_gdf.crs).explode(index_parts=False)

    return counties_dissolved_gdf


@pytest.fixture
def states_subset_path():
    """
    Returns path to states boundaries from states.geojson. This is a subset of
    states to be used for testing the make-maps CLI.
    """

    state_boundaries_path = Path(TEST_DATA_DIR).joinpath(
        "plots", "states.geojson"
    )

    return state_boundaries_path


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
def counties_gdf():
    """
    Return a geopandas geodataframe that is the counties boundaries from
    counties.geojson. To be used as the in
    utils.plots.map_geodataframe_column() tests.
    """

    county_boundaries_path = Path(TEST_DATA_DIR).joinpath(
        "plots", "counties.geojson"
    )
    counties_gdf = gpd.read_file(county_boundaries_path)
    counties_gdf.columns = [s.lower() for s in counties_gdf.columns]
    counties_gdf["cnty_fips"] = counties_gdf["cnty_fips"].astype(int)

    return counties_gdf


@pytest.fixture
def supply_curve_gdf():
    """
    Return a geopandas geodataframe of points from a test supply curve
    consisting of results for just a few states.
    """

    supply_curve_path = Path(TEST_DATA_DIR).joinpath(
        "plots", "map-supply-curve-solar.csv"
    )
    supply_curve_df = pd.read_csv(supply_curve_path)
    supply_curve_gdf = gpd.GeoDataFrame(
        supply_curve_df,
        geometry=gpd.points_from_xy(
            x=supply_curve_df['longitude'], y=supply_curve_df['latitude']
        ),
        crs="EPSG:4326"
    )

    return supply_curve_gdf


@pytest.fixture
def output_map_names():
    """
    Returns expected output map names (excluding file extensions) by technology
    for make-maps CLI tests
    """

    map_names = {
        "wind": [
            "area_sq_km_wind",
            "capacity_density_wind",
            "capacity_mw_wind",
            "lcot_wind",
            "mean_lcoe_wind",
            "total_lcoe_wind",
        ],
        "solar": [
            "area_sq_km_solar",
            "capacity_density_solar",
            "capacity_mw_ac_solar",
            "capacity_mw_dc_solar",
            "lcot_solar",
            "mean_lcoe_solar",
            "total_lcoe_solar",
        ]
    }

    return map_names


@pytest.fixture
def histogram_plot_area_sq_km():
    """Returns contents of ASCII histogram with default 20 bins for area_sq_km
    column of map_supply_curve_wind"""

    ascii_histogram_contents_src = Path(TEST_DATA_DIR).joinpath(
        "plots", "histogram_area_sq_km.txt"
    )
    with open(ascii_histogram_contents_src, "r", encoding="utf-8") as f:
        contents = f.read()

    return contents


@pytest.fixture
def histogram_plot_capacity_mw():
    """Returns contents of ASCII histogram with default 20 bins for
    capacity_mw column of map_supply_curve_wind"""

    ascii_histogram_contents_src = Path(TEST_DATA_DIR).joinpath(
        "plots", "histogram_capacity_mw.txt"
    )
    with open(ascii_histogram_contents_src, "r", encoding="utf-8") as f:
        contents = f.read()

    return contents


@pytest.fixture
def histogram_plot_area_sq_km_5bins():
    """Returns contents of ASCII histogram with 5 bins for area_sq_km column
    of map_supply_curve_wind"""

    ascii_histogram_contents_src = Path(TEST_DATA_DIR).joinpath(
        "plots", "histogram_area_sq_km_5bins.txt"
    )
    with open(ascii_histogram_contents_src, "r", encoding="utf-8") as f:
        contents = f.read()

    return contents


@pytest.fixture
def histogram_plot_capacity_mw_5bins():
    """Returns contents of ASCII histogram with 5 bins for capacity_mw column
    of map_supply_curve_wind"""

    ascii_histogram_contents_src = Path(TEST_DATA_DIR).joinpath(
        "plots", "histogram_capacity_mw_5bins.txt"
    )
    with open(ascii_histogram_contents_src, "r", encoding="utf-8") as f:
        contents = f.read()

    return contents


@pytest.fixture
def compare_images_approx():
    """Exposes the compare_images_approx function as a fixture"""
    return helper.compare_images_approx


def pytest_setup_options():
    """Recommended setup based on https://dash.plotly.com/testing."""
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--headless")
    return options
