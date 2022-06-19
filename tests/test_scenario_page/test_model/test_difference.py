# -*- coding: utf-8 -*-
"""Scenario Model tests."""
import pandas as pd

from reView.pages.rev.model import Difference
from reView.utils.functions import common_numeric_columns


def test_difference_calc(test_data_dir):
    """Test `Difference` class calc method."""

    diff = Difference('sc_point_gid')
    df1 = pd.read_csv(
        test_data_dir / 'hydrogen' / 'sample_data' / 'scenario_0.csv'
    )
    df2 = pd.read_csv(
        test_data_dir / 'hydrogen' / 'sample_data' / 'scenario_1.csv'
    )
    common_cols = common_numeric_columns(df1, df2)

    difference = diff.calc(df1, df2, "capacity")

    assert all(c in difference for c in df1.columns)
    assert all(c in difference for c in common_cols)
