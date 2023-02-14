# -*- coding: utf-8 -*-
"""CLI tests."""
import pathlib
import tempfile
import pytest
import geopandas as gpd
from reView.cli import (
    main,
    unpack_turbines
)


def test_main(test_cli_runner):
    """Test main() CLI command."""
    result = test_cli_runner.invoke(main)
    assert result.exit_code == 0


def test_unpack_turbines_happy(
    test_bespoke_supply_curve, test_cli_runner
):
    """Happy-path test for unpack_turbines() CLI command."""

    with tempfile.TemporaryDirectory() as td:
        output_gpkg = pathlib.Path(td).joinpath("bespoke.gpkg")
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

    with tempfile.TemporaryDirectory() as td:
        output_gpkg = pathlib.Path(td).joinpath("bespoke.gpkg")
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

    with tempfile.TemporaryDirectory() as td:
        output_gpkg = pathlib.Path(td).joinpath("bespoke.gpkg")
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

    with tempfile.TemporaryDirectory() as td:
        output_gpkg = pathlib.Path(td).joinpath("bespoke.gpkg")
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

    correct_results_gpkg = test_data_dir.joinpath('bespoke-supply-curve-turbines.gpkg')
    correct_df = gpd.read_file(correct_results_gpkg)

    with tempfile.TemporaryDirectory() as td:
        output_gpkg = pathlib.Path(td).joinpath("bespoke.gpkg")
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
    assert (correct_df == output_df).all().all(), \
        f"Output results do not match {correct_results_gpkg}"

if __name__ == '__main__':
    pytest.main([__file__, '-s'])
