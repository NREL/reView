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
    unpack_characterizations,
    make_maps,
    map_column
)
from tests.test_utils.test_plots import compare_images_approx


def test_main(cli_runner):
    """Test main() CLI command."""
    result = cli_runner.invoke(main)
    assert result.exit_code == 0


def test_unpack_turbines_happy(
    bespoke_supply_curve, cli_runner
):
    """Happy-path test for unpack_turbines() CLI command."""

    with tempfile.TemporaryDirectory() as tempdir:
        output_gpkg = pathlib.Path(tempdir).joinpath("bespoke.gpkg")
        result = cli_runner.invoke(
            unpack_turbines, [
                '-i', bespoke_supply_curve.as_posix(),
                '-o', output_gpkg,
                '-n', 1
            ]
        )
        assert result.exit_code == 0


def test_unpack_turbines_parallel(
    bespoke_supply_curve, cli_runner
):
    """Test unpack_turbines() CLI command with parallel processing."""

    with tempfile.TemporaryDirectory() as tempdir:
        output_gpkg = pathlib.Path(tempdir).joinpath("bespoke.gpkg")
        result = cli_runner.invoke(
            unpack_turbines, [
                '-i', bespoke_supply_curve.as_posix(),
                '-o', output_gpkg,
                '-n', 2
            ]
        )
        assert result.exit_code == 0


def test_unpack_turbines_no_overwrite(
    bespoke_supply_curve, cli_runner
):
    """Test unpack_turbines() CLI command correctly
        raises FileExistsError when output geopackage exists
        and overwrite flag is not used."""

    with tempfile.TemporaryDirectory() as tempdir:
        output_gpkg = pathlib.Path(tempdir).joinpath("bespoke.gpkg")
        with open(output_gpkg, 'wb'):
            pass
        result = cli_runner.invoke(
            unpack_turbines, [
                '-i', bespoke_supply_curve.as_posix(),
                '-o', output_gpkg,
                '-n', 1
            ]
        )
        assert result.exit_code == 1
        assert isinstance(result.exception, FileExistsError)


def test_unpack_turbines_overwrite(
    bespoke_supply_curve, cli_runner
):
    """Test unpack_turbines() CLI command correctly
        overwrites when output geopackage exists
        and overwrite flag is used."""

    with tempfile.TemporaryDirectory() as tempdir:
        output_gpkg = pathlib.Path(tempdir).joinpath("bespoke.gpkg")
        with open(output_gpkg, 'wb'):
            pass
        result = cli_runner.invoke(
            unpack_turbines, [
                '-i', bespoke_supply_curve.as_posix(),
                '-o', output_gpkg,
                '-n', 1,
                '--overwrite'
            ]
        )
        assert result.exit_code == 0


def test_unpack_turbines_results(
    bespoke_supply_curve, cli_runner, data_dir_test
):
    """Test that the data produced by unpack_turbines() CLI
        command matches known output file."""

    correct_results_gpkg = data_dir_test.joinpath(
        'bespoke-supply-curve-turbines.gpkg')
    correct_df = gpd.read_file(correct_results_gpkg)

    with tempfile.TemporaryDirectory() as tempdir:
        output_gpkg = pathlib.Path(tempdir).joinpath("bespoke.gpkg")
        result = cli_runner.invoke(
            unpack_turbines, [
                '-i', bespoke_supply_curve.as_posix(),
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
    characterization_supply_curve, cli_runner, data_dir_test
):
    """Test that the data produced by unpack_characterizations() CLI
       command matches known output file."""

    char_map_path = data_dir_test.joinpath("characterization-map.json")

    correct_results_src = data_dir_test.joinpath(
        'unpacked-characterization-supply-curve.csv')
    correct_df = pd.read_csv(correct_results_src)

    with tempfile.TemporaryDirectory() as tempdir:
        output_csv = pathlib.Path(tempdir).joinpath("characterizations.csv")
        result = cli_runner.invoke(
            unpack_characterizations, [
                '-i', characterization_supply_curve.as_posix(),
                '-m', char_map_path.as_posix(),
                '-o', output_csv
            ]
        )
        assert result.exit_code == 0

        output_df = pd.read_csv(output_csv)

    assert_frame_equal(output_df, correct_df)


@pytest.mark.filterwarnings("ignore:Skipping")
def test_unpack_characterizations_overwrite(
    characterization_supply_curve, cli_runner, data_dir_test
):
    """Test unpack_characterizations() CLI command correctly overwrites when
        output CSV exists and overwrite flag is used."""

    char_map_path = data_dir_test.joinpath("characterization-map.json")

    with tempfile.TemporaryDirectory() as tempdir:
        output_csv = pathlib.Path(tempdir).joinpath("characterizations.csv")
        with open(output_csv, 'w'):
            pass
        result = cli_runner.invoke(
            unpack_characterizations, [
                '-i', characterization_supply_curve.as_posix(),
                '-m', char_map_path.as_posix(),
                '-o', output_csv,
                '--overwrite'
            ]
        )
        assert result.exit_code == 0


@pytest.mark.filterwarnings("ignore:Skipping")
def test_unpack_characterizations_no_overwrite(
    characterization_supply_curve, cli_runner, data_dir_test
):
    """
    Test unpack_characterizations() CLI command correctly raises
    FileExistsError whe output CSV exists and overwrite flag is not used.
    """

    char_map_path = data_dir_test.joinpath("characterization-map.json")

    with tempfile.TemporaryDirectory() as tempdir:
        output_csv = pathlib.Path(tempdir).joinpath("characterizations.csv")
        with open(output_csv, 'w'):
            pass
        result = cli_runner.invoke(
            unpack_characterizations, [
                '-i', characterization_supply_curve.as_posix(),
                '-m', char_map_path.as_posix(),
                '-o', output_csv
            ]
        )
        assert result.exit_code == 1
        assert isinstance(result.exception, FileExistsError)


@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_make_maps_solar(
    map_supply_curve_solar, cli_runner, data_dir_test
):
    """
    Happy path test for make_maps() CLI. Tests that it produces the expected
    images for a solar supply curve.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_solar.as_posix(),
                '-t', "solar",
                '-o', output_path.as_posix(),
                '--dpi', 75
            ]
        )
        assert result.exit_code == 0

        out_png_names = [
            "capacity_solar.png",
            "lcot_solar.png",
            "mean_lcoe_solar.png",
            "total_lcoe_solar.png"
        ]
        for out_png_name in out_png_names:
            expected_png = data_dir_test.joinpath("plots", out_png_name)
            out_png = output_path.joinpath(out_png_name)
            assert compare_images_approx(expected_png, out_png), \
                "Output image does not match expected image " \
                f"{expected_png}"


@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in ")
def test_make_maps_wind(
    map_supply_curve_wind, cli_runner, data_dir_test
):
    """
    Happy path test for make_maps() CLI. Tests that it produces the expected
    images for a wind supply curve.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_wind.as_posix(),
                '-t', "wind",
                '-o', output_path.as_posix(),
                '--dpi', 75
            ]
        )
        assert result.exit_code == 0

        out_png_names = [
            "capacity_wind.png",
            "lcot_wind.png",
            "mean_lcoe_wind.png",
            "total_lcoe_wind.png",
            "capacity_density_wind.png",
        ]
        for out_png_name in out_png_names:
            expected_png = data_dir_test.joinpath("plots", out_png_name)
            out_png = output_path.joinpath(out_png_name)
            assert compare_images_approx(expected_png, out_png), \
                "Output image does not match expected image " \
                f"{expected_png}"


@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in ")
def test_make_maps_wind_keep_zero(
    map_supply_curve_wind, cli_runner, data_dir_test
):
    """
    Test that make_maps() CLI produces the expected images for a wind supply
    curve when the --keep-zero flag is enabled.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_wind.as_posix(),
                '-t', "wind",
                '-o', output_path.as_posix(),
                '--dpi', 75,
                '--keep-zero'
            ]
        )
        assert result.exit_code == 0

        out_png_names = [
            "capacity_wind.png",
            "lcot_wind.png",
            "mean_lcoe_wind.png",
            "total_lcoe_wind.png",
            "capacity_density_wind.png",
        ]
        for out_png_name in out_png_names:
            expected_png = data_dir_test.joinpath(
                "plots", out_png_name.replace(".png", "_kz.png")
            )
            out_png = output_path.joinpath(out_png_name)
            assert compare_images_approx(expected_png, out_png), \
                "Output image does not match expected image " \
                f"{expected_png}"


@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_make_maps_boundaries(
    map_supply_curve_solar, cli_runner, data_dir_test,
    states_subset_path
):
    """
    Test that make_maps() CLI works with an input boundaries file.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_solar.as_posix(),
                '-t', "solar",
                '-b', states_subset_path.as_posix(),
                '-o', output_path.as_posix(),
                '--dpi', 75
            ]
        )
        assert result.exit_code == 0

        out_png_names = [
            "capacity_solar.png",
            "lcot_solar.png",
            "mean_lcoe_solar.png",
            "total_lcoe_solar.png"
        ]
        for out_png_name in out_png_names:
            expected_png = data_dir_test.joinpath(
                "plots", out_png_name.replace(".png", "_boundaries.png")
            )
            out_png = output_path.joinpath(out_png_name)
            assert compare_images_approx(expected_png, out_png), \
                "Output image does not match expected image " \
                f"{expected_png}"


@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_column_happy(
    map_supply_curve_solar, cli_runner, data_dir_test
):
    """
    Happy path test for map_column() CLI. Tests that it produces the expected
    image for a solar supply curve and the area_sq_km column.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            map_column, [
                '-i', map_supply_curve_solar.as_posix(),
                '-c', 'area_sq_km',
                '-o', output_path.as_posix(),
                '--dpi', 75
            ]
        )
        assert result.exit_code == 0

        out_png_name = "area_sq_km.png"
        expected_png = data_dir_test.joinpath(
            "plots", out_png_name.replace(".png", "_happy.png")
        )
        out_png = output_path.joinpath(out_png_name)
        assert compare_images_approx(expected_png, out_png), \
            "Output image does not match expected image " \
            f"{expected_png}"


@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_column_formatting(
    map_supply_curve_solar, cli_runner, data_dir_test
):
    """
    Test that map_column() CLI produces the expected image when passed
    formatting options.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            map_column, [
                '-i', map_supply_curve_solar.as_posix(),
                '-c', 'area_sq_km',
                '-C', 'Greens',
                '-T', 'Developable Area (sq. km.)',
                '-B', '[10, 20, 30, 40]',
                '-o', output_path.as_posix(),
                '--dpi', 75
            ]
        )
        assert result.exit_code == 0

        out_png_name = "area_sq_km.png"
        expected_png = data_dir_test.joinpath(
            "plots", out_png_name.replace(".png", "_formatting.png")
        )
        out_png = output_path.joinpath(out_png_name)
        assert compare_images_approx(expected_png, out_png), \
            "Output image does not match expected image " \
            f"{expected_png}"


@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_column_bad_breaks(
    map_supply_curve_solar, cli_runner
):
    """
    Test that map_column() CLI raises a ValueError for badly formed inputs for
    legend_breaks.
    """
    bad_breaks = [
        '10, 20, 30, 40]',
        '[10, 20, 30, 40',
        '[10,20 30,40]',
        "not-breaks",
        10
    ]
    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        for breaks in bad_breaks:
            result = cli_runner.invoke(
                map_column, [
                    '-i', map_supply_curve_solar.as_posix(),
                    '-c', 'area_sq_km',
                    '-B', breaks,
                    '-o', output_path.as_posix(),
                    '--dpi', 75
                ]
            )
            assert result.exit_code == 1
            assert isinstance(result.exception, ValueError)


@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_column_bad_column(
    map_supply_curve_solar, cli_runner
):
    """
    Test that map_column() CLI raises a KeyError for an input column that
    doesn't exist in the supply curve dataset.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            map_column, [
                '-i', map_supply_curve_solar.as_posix(),
                '-c', 'not-a-column',
                '-o', output_path.as_posix(),
                '--dpi', 75
            ]
        )
        assert result.exit_code == 1
        assert isinstance(result.exception, KeyError)


@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_column_boundaries(
    map_supply_curve_solar, cli_runner, data_dir_test,
    states_subset_path
):
    """
    Test that map_column() CLI works with an input boundaries file.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            map_column, [
                '-i', map_supply_curve_solar.as_posix(),
                '-c', 'area_sq_km',
                '-o', output_path.as_posix(),
                '-b', states_subset_path,
                '--dpi', 75
            ]
        )
        assert result.exit_code == 0

        out_png_name = "area_sq_km.png"
        expected_png = data_dir_test.joinpath(
            "plots", out_png_name.replace(".png", "_boundaries.png")
        )
        out_png = output_path.joinpath(out_png_name)
        assert compare_images_approx(expected_png, out_png), \
            "Output image does not match expected image " \
            f"{expected_png}"


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
