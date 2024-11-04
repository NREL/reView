# -*- coding: utf-8 -*-
"""CLI tests."""
# pylint: disable=too-many-lines
import pathlib
import tempfile
from difflib import SequenceMatcher

import pytest
import click
import pandas as pd
import geopandas as gpd
from pandas.testing import assert_frame_equal
from reView.cli import (
    main,
    unpack_turbines,
    unpack_characterizations,
    make_maps,
    map_column,
    histogram,
    TECH_CHOICES
)


def test_main(cli_runner):
    """Test main() CLI command."""
    result = cli_runner.invoke(main)
    assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )


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
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )


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
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )


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
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )


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
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

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
    assert correct_df.geom_equals_exact(
        output_df, tolerance=1e-6, align=True
    ).all(), "Geometries are not the same."


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
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

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
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )


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


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_make_maps(
    map_supply_curve_wind, map_supply_curve_solar, cli_runner, data_dir_test,
    output_map_names, compare_images_approx
):
    """
    Happy path test for make_maps() CLI. Tests that it produces the expected
    images for both wind and solar supply curves.
    """

    for tech in TECH_CHOICES:
        if tech == "wind":
            supply_curve_file = map_supply_curve_wind.as_posix()
        else:
            supply_curve_file = map_supply_curve_solar.as_posix()

        with tempfile.TemporaryDirectory() as tempdir:
            output_path = pathlib.Path(tempdir)
            result = cli_runner.invoke(
                make_maps, [
                    '-i', supply_curve_file,
                    '-S', tech,
                    '-o', output_path.as_posix(),
                    '--dpi', 75
                ]
            )
            assert result.exit_code == 0, (
                f"Command failed with error {result.exception}"
            )

            out_png_names = [
                f"{mapname}.png" for mapname in output_map_names[tech]
            ]

            for out_png_name in out_png_names:
                expected_png = data_dir_test.joinpath("plots", out_png_name)
                out_png = output_path.joinpath(out_png_name)
                images_match, pct_diff = compare_images_approx(
                    expected_png, out_png
                )
                assert images_match, (
                    f"Output image does not match expected image "
                    f"{expected_png}. Difference is {pct_diff * 100}%"
                )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_make_maps_wind_keep_zero(
    map_supply_curve_wind_zeros, cli_runner, data_dir_test, output_map_names,
    compare_images_approx
):
    """
    Test that make_maps() CLI produces the expected images for a wind supply
    curve when the --keep-zero flag is enabled. Tests a mocked wind csv because
    normal post-processed CSVs do not have sites with zero capacity.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_wind_zeros.as_posix(),
                '-t', "wind",
                '-o', output_path.as_posix(),
                '--dpi', 75,
                '--keep-zero'
            ]
        )
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_png_names = [
            f"{mapname}.png" for mapname in output_map_names["wind"]
        ]
        for out_png_name in out_png_names:
            expected_png = data_dir_test.joinpath(
                "plots", out_png_name.replace(".png", "_kz.png")
            )
            out_png = output_path.joinpath(out_png_name)
            images_match, pct_diff = compare_images_approx(
                expected_png, out_png
            )
            assert images_match, (
                f"Output image does not match expected image {expected_png}"
                f"Difference is {pct_diff * 100}%"
            )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_make_maps_wind_drop_zero(
    map_supply_curve_wind_zeros, cli_runner, data_dir_test, output_map_names,
    compare_images_approx
):
    """
    Test that make_maps() CLI produces the expected images for a wind supply
    curve when the --keep-zero flag is NOT enabled. Tests a mocked wind csv
    because normal post-processed CSVs do not have sites with zero capacity.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_wind_zeros.as_posix(),
                '-t', "wind",
                '-o', output_path.as_posix(),
                '--dpi', 75
            ]
        )
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_png_names = [
            f"{mapname}.png" for mapname in output_map_names["wind"]
        ]
        for out_png_name in out_png_names:
            expected_png = data_dir_test.joinpath(
                "plots", out_png_name.replace(".png", "_dz.png")
            )
            out_png = output_path.joinpath(out_png_name)
            images_match, pct_diff = compare_images_approx(
                expected_png, out_png, max_diff_pct=0.34
            )
            assert images_match, (
                f"Output image does not match expected image {expected_png}"
                f"Difference is {pct_diff * 100}%"
            )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_make_maps_wind_drop_legend(
    map_supply_curve_wind, cli_runner, data_dir_test, output_map_names,
    compare_images_approx
):
    """
    Test that make_maps() CLI produces the expected images without legends
    when --drop-legend is specified.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_wind.as_posix(),
                '-t', "wind",
                '-o', output_path.as_posix(),
                '--dpi', 75,
                '--drop-legend'
            ]
        )
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_png_names = [
            f"{mapname}.png" for mapname in output_map_names["wind"]
        ]
        for out_png_name in out_png_names:
            expected_png = data_dir_test.joinpath(
                "plots", out_png_name.replace(".png", "_drop_legend.png")
            )
            out_png = output_path.joinpath(out_png_name)
            images_match, pct_diff = compare_images_approx(
                expected_png, out_png
            )
            assert images_match, (
                f"Output image does not match expected image {expected_png}"
                f"Difference is {pct_diff * 100}%"
            )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_make_maps_boundaries(
    map_supply_curve_solar, cli_runner, data_dir_test,
    states_subset_path, output_map_names, compare_images_approx
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
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_png_names = [
            f"{mapname}.png" for mapname in output_map_names["solar"]
        ]
        for out_png_name in out_png_names:
            expected_png = data_dir_test.joinpath(
                "plots", out_png_name.replace(".png", "_boundaries.png")
            )
            out_png = output_path.joinpath(out_png_name)
            images_match, pct_diff = compare_images_approx(
                expected_png, out_png, max_diff_pct=0.34
            )
            assert images_match, (
                f"Output image does not match expected image {expected_png}"
                f"Difference is {pct_diff * 100}%"
            )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_make_maps_wind_mc_breaks(
    map_supply_curve_wind, cli_runner, data_dir_test, output_map_names,
    compare_images_approx
):
    """
    Test that make_maps() CLI produces the expected images when a mapclassifier
    scheme is passed as input to the --breaks-scheme parameter
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_wind.as_posix(),
                '-S', "equalinterval:{\"k\": 6}",
                '-o', output_path.as_posix(),
                '--dpi', 75
            ]
        )
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_png_names = [
            f"{mapname.replace('wind', 'equalinterval')}.png" for mapname
            in output_map_names["wind"]
        ]
        for out_png_name in out_png_names:
            expected_png = data_dir_test.joinpath("plots", out_png_name)
            out_png = output_path.joinpath(out_png_name)
            images_match, pct_diff = compare_images_approx(
                expected_png, out_png
            )
            assert images_match, (
                f"Output image does not match expected image {expected_png}"
                f"Difference is {pct_diff * 100}%"
            )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_make_maps_wind_bad_breaks_scheme(map_supply_curve_wind, cli_runner):
    """
    Test that make_maps() CLI raises BadParameter exceptions when the inputs
    to --breaks-scheme are not valid.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)

        # invalid classifier name
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_wind.as_posix(),
                '-S', "fakeclassifier:{\"k\": 6}",
                '-o', output_path.as_posix(),
                '--dpi', 75
            ], standalone_mode=False
        )
        assert result.exit_code == 1, "Command did not fail as expected"
        with pytest.raises(
            click.BadParameter,
            match="Classifier fakeclassifier not recognized*"
        ):
            raise result.exception

        # kwargs are not valid json
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_wind.as_posix(),
                '-S', "equalinterval:\"k\": 6",
                '-o', output_path.as_posix(),
                '--dpi', 75
            ], standalone_mode=False
        )
        assert result.exit_code == 1, "Command did not fail as expected"
        with pytest.raises(
            click.BadParameter,
            match="Keyword arguments for classifier must be formated as valid "
            "JSON."
        ):
            raise result.exception

        # neither tech nor breaks-scheme specified
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_wind.as_posix(),
                '-o', output_path.as_posix(),
                '--dpi', 75
            ], standalone_mode=False
        )
        assert result.exit_code == 1, "Command did not fail as expected"
        with pytest.raises(
            click.BadParameter,
            match="Either --breaks-scheme or --tech must be specified."
        ):
            raise result.exception


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_make_maps_wind_breaks_and_techs(
    map_supply_curve_wind, cli_runner, data_dir_test, output_map_names,
    compare_images_approx
):
    """
    Test that make_maps() CLI produces the expected images when both --tech and
    --breaks-scheme are specified
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        with pytest.warns(
            UserWarning,
            match="Both --breaks-scheme and --tech were specified*"
        ):
            result = cli_runner.invoke(
                make_maps, [
                    '-i', map_supply_curve_wind.as_posix(),
                    '-S', "wind",
                    '-t', "equalinterval:{\"k\": 6}",
                    '-o', output_path.as_posix(),
                    '--dpi', 75
                ], standalone_mode=True
            )
            assert result.exit_code == 0, (
                f"Command failed with error {result.exception}"
            )

        out_png_names = [
            f"{mapname}.png" for mapname in output_map_names["wind"]
        ]
        for out_png_name in out_png_names:
            expected_png = data_dir_test.joinpath("plots", out_png_name)
            out_png = output_path.joinpath(out_png_name)
            images_match, pct_diff = compare_images_approx(
                expected_png, out_png
            )
            assert images_match, (
                f"Output image does not match expected image {expected_png}"
                f"Difference is {pct_diff * 100}%"
            )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_make_maps_pdf(map_supply_curve_solar, cli_runner, output_map_names):
    """
    Test that make_maps() CLI correctly outputs pdfs when specified as the
    out_format.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            make_maps, [
                '-i', map_supply_curve_solar.as_posix(),
                '-t', "solar",
                '-o', output_path.as_posix(),
                '--dpi', 75,
                '-F', 'pdf'
            ]
        )
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_pdf_names = [
            f"{mapname}.pdf" for mapname in output_map_names["solar"]
        ]
        # it's not easy to open and check pdf contents, so just check
        # the file was created successfully and is not empty
        for out_pdf_name in out_pdf_names:
            out_pdf = output_path.joinpath(out_pdf_name)
            assert out_pdf.exists(), \
                f"Output pdf could not be found: {out_pdf_name}"
            assert out_pdf.stat().st_size > 0, \
                f"Output pdf appears to be an empty file: {out_pdf_name}"


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_column_happy(
    map_supply_curve_solar, cli_runner, data_dir_test, compare_images_approx
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
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_png_name = "area_sq_km.png"
        expected_png = data_dir_test.joinpath(
            "plots", out_png_name.replace(".png", "_happy.png")
        )
        out_png = output_path.joinpath(out_png_name)
        images_match, pct_diff = compare_images_approx(
            expected_png, out_png
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_column_formatting(
    map_supply_curve_solar, cli_runner, data_dir_test, compare_images_approx
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
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_png_name = "area_sq_km.png"
        expected_png = data_dir_test.joinpath(
            "plots", out_png_name.replace(".png", "_formatting.png")
        )
        out_png = output_path.joinpath(out_png_name)
        images_match, pct_diff = compare_images_approx(
            expected_png, out_png
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.maptest
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


@pytest.mark.maptest
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


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_column_boundaries(
    map_supply_curve_solar, cli_runner, data_dir_test,
    states_subset_path, compare_images_approx
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
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_png_name = "area_sq_km.png"
        expected_png = data_dir_test.joinpath(
            "plots", out_png_name.replace(".png", "_boundaries.png")
        )
        out_png = output_path.joinpath(out_png_name)
        images_match, pct_diff = compare_images_approx(
            expected_png, out_png
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_column_boundaries_kwargs(
    map_supply_curve_solar, cli_runner, data_dir_test, compare_images_approx
):
    """
    Test that map_column() CLI works when boundaries_kwargs are specified
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            map_column, [
                '-i', map_supply_curve_solar.as_posix(),
                '-c', 'area_sq_km',
                '-o', output_path.as_posix(),
                '-bk', '{"linewidth": 1, "zorder": 2, "edgecolor": "black"}',
                '--dpi', 75
            ]
        )
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_png_name = "area_sq_km.png"
        expected_png = data_dir_test.joinpath(
            "plots", out_png_name.replace(".png", "_boundaries_kwargs.png")
        )
        out_png = output_path.joinpath(out_png_name)
        images_match, pct_diff = compare_images_approx(
            expected_png, out_png
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_column_drop_legend(
    map_supply_curve_solar, cli_runner, data_dir_test, compare_images_approx
):
    """
    Test that map_column() CLI works when --drop-legend is specified
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            map_column, [
                '-i', map_supply_curve_solar.as_posix(),
                '-c', 'area_sq_km',
                '-o', output_path.as_posix(),
                '--drop-legend',
                '--dpi', 75
            ]
        )
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_png_name = "area_sq_km.png"
        expected_png = data_dir_test.joinpath(
            "plots", out_png_name.replace(".png", "_drop_legend.png")
        )
        out_png = output_path.joinpath(out_png_name)
        images_match, pct_diff = compare_images_approx(
            expected_png, out_png
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Skipping")
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_column_pdf(map_supply_curve_solar, cli_runner):
    """
    Test that map_column() CLI correctly outputs a pdf when specified as the
    out-format.
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = pathlib.Path(tempdir)
        result = cli_runner.invoke(
            map_column, [
                '-i', map_supply_curve_solar.as_posix(),
                '-c', 'area_sq_km',
                '-o', output_path.as_posix(),
                '--dpi', 75,
                '-F', 'pdf'
            ]
        )
        assert result.exit_code == 0, (
            f"Command failed with error {result.exception}"
        )

        out_pdf_name = "area_sq_km.pdf"
        out_pdf = output_path.joinpath(out_pdf_name)
        assert out_pdf.exists(), \
            f"Output pdf could not be found: {out_pdf_name}"
        assert out_pdf.stat().st_size > 0, \
            f"Output pdf appears to be an empty file: {out_pdf_name}"


def test_histogram_default_bins(
    cli_runner, map_supply_curve_wind, histogram_plot_area_sq_km,
    histogram_plot_capacity_mw
):
    """
    Test that histogram command correctly plots two variables and adjusts
    the plot size to the input height and width
    """

    result = cli_runner.invoke(
        histogram, [
            map_supply_curve_wind.as_posix(),
            '-c', 'area_sq_km',
            '-c', 'capacity_mw',
            '-W', 75,
            '-H', 15
        ],
        terminal_width=1000  # needed for height to be applied correctly
    )
    assert result.exit_code == 0, (
        f"Command failed with error {result.exception}"
    )

    plot = result.output
    expected_combined_plot = (
        histogram_plot_area_sq_km + '\n' + histogram_plot_capacity_mw
    )
    similarity = SequenceMatcher(None, plot, expected_combined_plot).ratio()
    assert similarity >= 0.95, (
        "ASCII Histogram does not match expected result: "
        f"similarity is only: {similarity}"
    )


def test_histogram_5bins(
    cli_runner, map_supply_curve_wind, histogram_plot_area_sq_km_5bins,
    histogram_plot_capacity_mw_5bins
):
    """
    Test that histogram command correctly adjusts based on a specified
    number of bins.
    """

    result = cli_runner.invoke(
        histogram, [
            map_supply_curve_wind.as_posix(),
            '-c', 'area_sq_km',
            '-c', 'capacity_mw',
            '-W', 75,
            '-H', 15,
            '-N', 5
        ],
        terminal_width=1000  # needed for height to be applied correctly
    )
    assert result.exit_code == 0, (
        f"Command failed with error {result.exception}"
    )

    plot = result.output
    expected_combined_plot = (
        histogram_plot_area_sq_km_5bins + '\n' +
        histogram_plot_capacity_mw_5bins
    )
    similarity = SequenceMatcher(None, plot, expected_combined_plot).ratio()
    assert similarity >= 0.95, (
        "ASCII Histogram does not match expected result: "
        f"similarity is only: {similarity}"
    )


def test_histogram_nonnumeric_column(
    cli_runner, map_supply_curve_wind, histogram_plot_area_sq_km
):
    """
    Test that histogram command gracefully handles a non-numeric input column
    by issuing a warning message and skipping over it.
    """

    result = cli_runner.invoke(
        histogram,
        [
            map_supply_curve_wind.as_posix(),
            '-c', 'state',
            '-c', 'area_sq_km',
            '-W', 75,
            '-H', 15
        ],
        terminal_width=1000  # needed for height to be applied correctly
    )
    assert result.exit_code == 0, (
        f"Command failed with error {result.exception}"
    )
    assert result.output.startswith(
        "Unable to plot column 'state': "
        "Input column must have a numeric dtype."
    )

    plot = result.output.split("\n", maxsplit=1)[1]
    similarity = SequenceMatcher(None, plot, histogram_plot_area_sq_km).ratio()
    assert similarity >= 0.95, (
        "ASCII Histogram does not match expected result: "
        f"similarity is only: {similarity}"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
