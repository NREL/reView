# -*- coding: utf-8 -*-
"""CLI tests."""
import pathlib
import tempfile
import pytest
import pandas as pd
import geopandas as gpd
from pandas.testing import assert_frame_equal
from reView.cli import (
    main,
    unpack_turbines,
    unpack_characterizations
)


def test_main(test_cli_runner):
    """Test main() CLI command."""
    result = test_cli_runner.invoke(main)
    assert result.exit_code == 0


def test_unpack_turbines_happy(
    test_bespoke_supply_curve, test_cli_runner
):
    """Happy-path test for unpack_turbines() CLI command."""

    with tempfile.TemporaryDirectory() as tempdir:
        output_gpkg = pathlib.Path(tempdir).joinpath("bespoke.gpkg")
        result = test_cli_runner.invoke(
            unpack_turbines, [
                '-i', test_bespoke_supply_curve.as_posix(),
                '-o', output_gpkg,
                '-n', 1
            ]
        )
        assert result.exit_code == 0


def test_unpack_turbines_parallel(
    test_bespoke_supply_curve, test_cli_runner
):
    """Test unpack_turbines() CLI command with parallel processing."""

    with tempfile.TemporaryDirectory() as tempdir:
        output_gpkg = pathlib.Path(tempdir).joinpath("bespoke.gpkg")
        result = test_cli_runner.invoke(
            unpack_turbines, [
                '-i', test_bespoke_supply_curve.as_posix(),
                '-o', output_gpkg,
                '-n', 2
            ]
        )
        assert result.exit_code == 0


def test_unpack_turbines_no_overwrite(
    test_bespoke_supply_curve, test_cli_runner
):
    """Test unpack_turbines() CLI command correctly
        raises FileExistsError when output geopackage exists
        and overwrite flag is not used."""

    with tempfile.TemporaryDirectory() as tempdir:
        output_gpkg = pathlib.Path(tempdir).joinpath("bespoke.gpkg")
        with open(output_gpkg, 'wb'):
            pass
        result = test_cli_runner.invoke(
            unpack_turbines, [
                '-i', test_bespoke_supply_curve.as_posix(),
                '-o', output_gpkg,
                '-n', 1
            ]
        )
        assert result.exit_code == 1
        assert isinstance(result.exception, FileExistsError)


def test_unpack_turbines_overwrite(
    test_bespoke_supply_curve, test_cli_runner
):
    """Test unpack_turbines() CLI command correctly
        overwrites when output geopackage exists
        and overwrite flag is used."""

    with tempfile.TemporaryDirectory() as tempdir:
        output_gpkg = pathlib.Path(tempdir).joinpath("bespoke.gpkg")
        with open(output_gpkg, 'wb'):
            pass
        result = test_cli_runner.invoke(
            unpack_turbines, [
                '-i', test_bespoke_supply_curve.as_posix(),
                '-o', output_gpkg,
                '-n', 1,
                '--overwrite'
            ]
        )
        assert result.exit_code == 0


def test_unpack_turbines_results(
    test_bespoke_supply_curve, test_cli_runner, test_data_dir
):
    """Test that the data produced by unpack_turbines() CLI
        command matches known output file."""

    correct_results_gpkg = test_data_dir.joinpath(
        'bespoke-supply-curve-turbines.gpkg')
    correct_df = gpd.read_file(correct_results_gpkg)

    with tempfile.TemporaryDirectory() as tempdir:
        output_gpkg = pathlib.Path(tempdir).joinpath("bespoke.gpkg")
        result = test_cli_runner.invoke(
            unpack_turbines, [
                '-i', test_bespoke_supply_curve.as_posix(),
                '-o', output_gpkg,
                '-n', 1
            ]
        )
        assert result.exit_code == 0

        output_df = gpd.read_file(output_gpkg)

    # this sort is unnecessary at the moment but for future-proofing
    correct_df.sort_values(by=['latitude', 'longitude'], inplace=True)
    output_df.sort_values(by=['latitude', 'longitude'], inplace=True)
    correct_df_no_geoms = pd.DataFrame(correct_df.drop(columns='geometry'))
    output_df_no_geoms = pd.DataFrame(output_df.drop(columns='geometry'))
    assert_frame_equal(
        correct_df_no_geoms,
        output_df_no_geoms,
        check_exact=False,
        rtol=0.001)
    assert correct_df.geom_almost_equals(output_df).all(),\
        "Geometries are not the same."


@pytest.mark.filterwarnings("ignore:Skipping")
def test_unpack_characterizations(
    test_characterization_supply_curve, test_cli_runner, test_data_dir
):
    """Test that the data produced by unpack_characterizations() CLI
       command matches known output file."""

    char_map_path = test_data_dir.joinpath("characterization-map.json")

    correct_results_src = test_data_dir.joinpath(
        'unpacked-characterization-supply-curve.csv')
    correct_df = pd.read_csv(correct_results_src)

    with tempfile.TemporaryDirectory() as tempdir:
        output_csv = pathlib.Path(tempdir).joinpath("characterizations.csv")
        result = test_cli_runner.invoke(
            unpack_characterizations, [
                '-i', test_characterization_supply_curve.as_posix(),
                '-m', char_map_path.as_posix(),
                '-o', output_csv
            ]
        )
        assert result.exit_code == 0

        output_df = pd.read_csv(output_csv)

    assert_frame_equal(output_df, correct_df)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_unpack_characterizations_overwrite(
    test_characterization_supply_curve, test_cli_runner, test_data_dir
):
    """Test unpack_characterizations() CLI command correctly overwrites when
        output CSV exists and overwrite flag is used."""

    char_map_path = test_data_dir.joinpath("characterization-map.json")

    with tempfile.TemporaryDirectory() as tempdir:
        output_csv = pathlib.Path(tempdir).joinpath("characterizations.csv")
        with open(output_csv, 'w'):
            pass
        result = test_cli_runner.invoke(
            unpack_characterizations, [
                '-i', test_characterization_supply_curve.as_posix(),
                '-m', char_map_path.as_posix(),
                '-o', output_csv,
                '--overwrite'
            ]
        )
        assert result.exit_code == 0


@pytest.mark.filterwarnings("ignore:Skipping")
def test_unpack_characterizations_no_overwrite(
    test_characterization_supply_curve, test_cli_runner, test_data_dir
):
    """
    Test unpack_characterizations() CLI command correctly raises
    FileExistsError whe output CSV exists and overwrite flag is not used.
    """

    char_map_path = test_data_dir.joinpath("characterization-map.json")

    with tempfile.TemporaryDirectory() as tempdir:
        output_csv = pathlib.Path(tempdir).joinpath("characterizations.csv")
        with open(output_csv, 'w'):
            pass
        result = test_cli_runner.invoke(
            unpack_characterizations, [
                '-i', test_characterization_supply_curve.as_posix(),
                '-m', char_map_path.as_posix(),
                '-o', output_csv
            ]
        )
        assert result.exit_code == 1
        assert isinstance(result.exception, FileExistsError)


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
