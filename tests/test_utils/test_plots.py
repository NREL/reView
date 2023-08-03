# -*- coding: utf-8 -*-
"""Plots unit tests."""
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
import io

import pytest
import mapclassify as mc
import numpy as np
import matplotlib.pyplot as plt
import PIL
import imagehash
import pandas as pd


from reView.utils.plots import (
    YBFixedBounds, map_geodataframe_column, ascii_histogram
)


def compare_images_approx(
    image_1_path, image_2_path, hash_size=12, max_diff_pct=0.25
):
    """
    Check if two images match approximately.

    Parameters
    ----------
    image_1_path : pathlib.Path
        File path to first image.
    image_2_path : pathlib.Path
        File path to first image.
    hash_size : int, optional
        Size of the image hashes that will be used for image comparison,
        by default 12. Increase to make the check more precise, decrease to
        make it more approximate.
    max_diff_pct : float, optional
        Tolerance for the amount of difference allowed, by default 0.05 (= 5%).
        Increase to allow for a larger delta between the image hashes, decrease
        to make the check stricter and require a smaller delta between the
        image hashes.

    Returns
    -------
    bool
        Returns true if the images match approximately, false if not.
    """

    expected_hash = imagehash.phash(
        PIL.Image.open(image_1_path), hash_size=hash_size
    )
    out_hash = imagehash.phash(
        PIL.Image.open(image_2_path), hash_size=hash_size
    )

    max_diff_bits = int(np.ceil(hash_size * max_diff_pct))

    diff = expected_hash - out_hash
    matches = diff <= max_diff_bits
    pct_diff = float(diff) / hash_size

    return matches, pct_diff


def test_YBFixedBounds_happy():
    # pylint: disable=invalid-name
    """
    Happy path test for YBFixedBounds. Check that it correctly resets
    max() and min() methods to return preset values rather than actual min
    and max of the input array.
    """

    data = np.arange(1, 10)

    preset_max = 15
    preset_min = 0
    yb = YBFixedBounds(data, preset_max=15, preset_min=0)

    assert data.max() != preset_max
    assert data.min() != preset_min
    assert yb.max() == preset_max
    assert yb.min() == preset_min


def test_YBFixedBounds_mapclassify():
    # pylint: disable=invalid-name
    """
    Test YBFixedBounds works as expected when used to overwrite the yb
    property of a mapclassify classifier.
    """

    data = np.arange(1, 90)
    breaks = [20, 40, 60, 80, 100]
    scheme = mc.UserDefined(data, bins=breaks)
    preset_max = scheme.k
    present_min = 0

    assert scheme.yb.max() < scheme.k
    scheme.yb = YBFixedBounds(
        scheme.yb, preset_max=preset_max, preset_min=present_min
    )
    assert scheme.yb.max() == preset_max
    assert scheme.yb.min() == present_min


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_happy(
    data_dir_test, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Happy path test for map_geodataframe_column. Test that when run
    with basic inputs and default settings, the output image matches
    the expected image.
    """
    col_name = "area_sq_km"

    with tempfile.TemporaryDirectory() as tempdir:

        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf
        )
        plt.tight_layout()

        out_png_name = "happy_map.png"
        out_png = Path(tempdir).joinpath("happy_map.png")
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = data_dir_test.joinpath("plots", out_png_name)

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_styling(
    data_dir_test, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Test that map_geodataframe_column() produces expected output image when
    various styling parameters are passed.
    """

    col_name = "area_sq_km"
    color_map = "GnBu"

    breaks = [15, 20, 25, 30, 40]
    map_extent = states_gdf.buffer(0.05).total_bounds

    with tempfile.TemporaryDirectory() as tempdir:

        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            color_map=color_map,
            breaks=breaks,
            map_title="Styling Map",
            legend_title=col_name.title(),
            background_df=background_gdf,
            boundaries_df=states_gdf,
            extent=map_extent,
            layer_kwargs={"s": 4, "linewidth": 0, "marker": "o"},
            legend_kwargs={
                "marker": "o",
                "frameon": True,
                "bbox_to_anchor": (1, 0),
                "loc": "upper left"
            }
        )
        plt.tight_layout()

        out_png_name = "styling_map.png"
        out_png = Path(tempdir).joinpath(out_png_name)
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = data_dir_test.joinpath("plots", out_png_name)

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_repeat(
    data_dir_test, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Test that running map_geodataframe_column twice exactly the same produces
    the same output. This covers a previously discovered bug where the legend
    symbols would change from squares to circles for the second map in a
    sequence.
    """
    col_name = "area_sq_km"

    with tempfile.TemporaryDirectory() as tempdir:

        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf
        )
        plt.tight_layout()
        plt.close(g.figure)

        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf
        )
        plt.tight_layout()

        out_png_name = "happy_map.png"
        out_png = Path(tempdir).joinpath(out_png_name)
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = data_dir_test.joinpath("plots", out_png_name)

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_no_legend(
    data_dir_test, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Test that map_geodataframe_column function produces a map without a legend
    when the legend argument is specified as False.
    """
    col_name = "area_sq_km"

    with tempfile.TemporaryDirectory() as tempdir:

        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf,
            legend=False
        )
        plt.tight_layout()

        out_png_name = "no_legend.png"
        out_png = Path(tempdir).joinpath("no_legend.png")
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = data_dir_test.joinpath("plots", out_png_name)

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_boundaries_kwargs(
    data_dir_test, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Test that map_geodataframe_column function produces a map with correctly
    styled boundaries when boundaries_kwargs are passed.
    """
    col_name = "area_sq_km"

    with tempfile.TemporaryDirectory() as tempdir:

        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf,
            boundaries_kwargs={
                "linewidth": 2, "zorder": 1, "edgecolor": "black"
            }
        )
        plt.tight_layout()

        out_png_name = "boundaries_kwargs.png"
        out_png = Path(tempdir).joinpath("boundaries_kwargs.png")
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = data_dir_test.joinpath("plots", out_png_name)

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


@pytest.mark.maptest
@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_polygons(
    data_dir_test, supply_curve_gdf, county_background_gdf, states_gdf,
    counties_gdf
):
    """
    Test that map_geodataframe_column() produces expected output image
    for a polygon input layer.
    """

    county_area_sq_km_df = supply_curve_gdf.groupby(
        "cnty_fips"
    )["area_sq_km"].sum().reset_index()
    county_capacity_gdf = counties_gdf.merge(
        county_area_sq_km_df, how="inner", on="cnty_fips"
    )

    col_name = "area_sq_km"
    color_map = "YlOrRd"

    breaks = [250, 350, 450, 550]
    map_extent = county_background_gdf.buffer(0.05).total_bounds

    with tempfile.TemporaryDirectory() as tempdir:

        g = map_geodataframe_column(
            county_capacity_gdf,
            col_name,
            color_map=color_map,
            breaks=breaks,
            map_title="Polygons Map",
            legend_title=col_name.title(),
            background_df=county_background_gdf,
            boundaries_df=states_gdf,
            extent=map_extent,
            layer_kwargs={"edgecolor": "gray", "linewidth": 0.5},
            boundaries_kwargs={
                "linewidth": 1,
                "zorder": 2,
                "edgecolor": "black",
            },
            legend_kwargs={
                "marker": "s",
                "frameon": False,
                "bbox_to_anchor": (1, 0.5),
                "loc": "center left"
            }
        )
        plt.tight_layout()

        out_png_name = "polygons_map.png"
        out_png = Path(tempdir).joinpath(out_png_name)
        g.figure.savefig(out_png, dpi=75)
        plt.close(g.figure)

        expected_png = data_dir_test.joinpath("plots", out_png_name)

        images_match, pct_diff = compare_images_approx(
            expected_png, out_png
        )
        assert images_match, (
            f"Output image does not match expected image {expected_png}"
            f"Difference is {pct_diff * 100}%"
        )


def test_ascii_histogram(map_supply_curve_wind, ascii_histogram_contents):
    """Happy path unit test for the ascii_histogram function"""

    df = pd.read_csv(map_supply_curve_wind)

    f = io.StringIO()
    with redirect_stdout(f):
        ascii_histogram(df, "area_sq_km", width=75, height=50)
    plot = f.getvalue()

    assert plot == ascii_histogram_contents, \
        "ASCII Histogram does not match expected result"


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
