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
            # layer_kwargs={"s": 4, "linewidth": 0, "marker": "o"}
        )
        plt.tight_layout()

        out_png_name = "happy_map.png"
        out_png = Path(tempdir).joinpath("happy_map.png")
        g.figure.savefig(out_png, dpi=600)
        plt.close(g.figure)

        expected_png = test_data_dir.joinpath("plots", out_png_name)

        hash_size = 16
        expected_hash = imagehash.phash(
            PIL.Image.open(expected_png), hash_size=hash_size
        )
        out_hash = imagehash.phash(
            PIL.Image.open(out_png), hash_size=hash_size
        )
        max_diff_pct = 0.05
        max_diff_bits = int(np.ceil(hash_size * max_diff_pct))
        assert expected_hash - out_hash < max_diff_bits, \
            f"Output image does not match expected image {expected_png}"

        # out_image = np.asarray(PIL.Image.open(out_png))
        # expected_image = np.asarray(PIL.Image.open(expected_png))

        # assert np.all(out_image == expected_image), \
        #     f"Output image does not match expected image {expected_png}"



if __name__ == '__main__':
    pytest.main([__file__, '-s'])
