# -*- coding: utf-8 -*-
"""Plots unit tests."""
import tempfile
from pathlib import Path

import pytest
import mapclassify as mc
import numpy as np
import matplotlib.pyplot as plt
import PIL
import imagehash

from reView.utils.plots import YBFixedBounds, map_geodataframe_column


def compare_images_approx(
    image_1_path, image_2_path, hash_size=16, max_diff_pct=0.05
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
        by default 16. Increase to make the check more precise, decrease to
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

    hash_size = 16
    expected_hash = imagehash.phash(
        PIL.Image.open(image_1_path), hash_size=hash_size
    )
    out_hash = imagehash.phash(
        PIL.Image.open(image_2_path), hash_size=hash_size
    )

    max_diff_bits = int(np.ceil(hash_size * max_diff_pct))

    return expected_hash - out_hash < max_diff_bits


def compare_images_exact(image_1_path, image_2_path):
    """
    Check if two images match exactly.

    Parameters
    ----------
    image_1_path : pathlib.Path
        File path to first image.
    image_2_path : pathlib.Path
        File path to first image.

    Returns
    -------
    bool
        Returns true if the images match approximately, false if not.
    """

    image_1 = np.asarray(PIL.Image.open(image_1_path))
    image_2 = np.asarray(PIL.Image.open(image_2_path))

    return np.all(image_1 == image_2)


# pylint: disable=invalid-name
def test_YBFixedBounds_happy():
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


# pylint: disable=invalid-name
def test_YBFixedBounds_mapclassify():
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


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_happy(
    test_data_dir, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Happy path test for map_geodataframe_column. Test that when run
    with known inputs and default style settings, the output image matches
    the expected image.
    """
    col_name = "capacity"
    color_map = "GnBu"

    breaks = [500, 1000, 1500, 2000]
    map_extent = states_gdf.buffer(1.5).total_bounds

    with tempfile.TemporaryDirectory() as tempdir:

        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            color_map=color_map,
            breaks=breaks,
            map_title="Happy Map",
            legend_title=col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf,
            extent=map_extent
        )
        plt.tight_layout()

        out_png_name = "happy_map.png"
        out_png = Path(tempdir).joinpath("happy_map.png")
        g.figure.savefig(out_png, dpi=600)
        plt.close(g.figure)

        expected_png = test_data_dir.joinpath("plots", out_png_name)

        images_match_exactly = compare_images_exact(expected_png, out_png)
        if not images_match_exactly:
            assert compare_images_approx(expected_png, out_png), \
                f"Output image does not match expected image {expected_png}"


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic:UserWarning")
def test_map_geodataframe_column_styling(
    test_data_dir, supply_curve_gdf, background_gdf, states_gdf
):
    """
    Test that map_geodataframe_column() produces expected output image when
    various styling parameters are passed.
    """

    col_name = "capacity"
    color_map = "GnBu"

    breaks = [500, 1000, 1500, 2000]
    map_extent = states_gdf.buffer(1.5).total_bounds

    with tempfile.TemporaryDirectory() as tempdir:

        g = map_geodataframe_column(
            supply_curve_gdf,
            col_name,
            color_map=color_map,
            breaks=breaks,
            map_title="Happy Map",
            legend_title=col_name,
            background_df=background_gdf,
            boundaries_df=states_gdf,
            extent=map_extent,
            layer_kwargs={"s": 4, "linewidth": 0, "marker": "o"}
        )
        plt.tight_layout()

        out_png_name = "styling_map.png"
        out_png = Path(tempdir).joinpath("styling_map.png")
        g.figure.savefig(out_png, dpi=600)
        plt.close(g.figure)

        expected_png = test_data_dir.joinpath("plots", out_png_name)

        images_match_exactly = compare_images_exact(expected_png, out_png)
        if not images_match_exactly:
            assert compare_images_approx(expected_png, out_png), \
                f"Output image does not match expected image {expected_png}"

# TODO: add more style params to test_map_geodataframe_column_styling
# TODO: write a test for map_geodataframe_column that tests polygons
# TODO: add tests for a few other parameters of map_geodataframe_column
# TODO: add more detailed examples in the docs
# TODO: add cli with functionality for creating a standard set of maps for an input sc
# (e.g., LCOE, capacity, ??)
if __name__ == '__main__':
    pytest.main([__file__, '-s'])
