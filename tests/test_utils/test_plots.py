# -*- coding: utf-8 -*-
"""Plots unit tests."""
import pytest
import mapclassify as mc
import numpy as np

from reView.utils.plots import YBFixedBounds, map_geodataframe_column


def test_YBFixedBounds_happy():
    """
    Happy path test for YBFixedBounds. Check that it correctly resets
    max() and min() methods to return preset values rather than actual min
    and max of the input array.
    """

    a = np.arange(1, 10)

    preset_max = 15
    preset_min = 0
    yb = YBFixedBounds(a, preset_max=15, preset_min=0)

    assert a.max() != preset_max
    assert a.min() != preset_min
    assert yb.max() == preset_max
    assert yb.min() == preset_min


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


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
