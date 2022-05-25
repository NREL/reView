# -*- coding: utf-8 -*-
"""Scenario Model tests."""
from itertools import product

import pandas as pd

from reView.pages.scenario.scenario_data import Difference
from reView.utils.functions import common_numeric_columns
from reView.utils.classes import DiffUnitOptions


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

    difference = diff.calc(df1, df2)

    assert len(difference) == len(df1)
    assert all(c in difference for c in df1.columns)
    assert all(c in difference for c in common_cols)

    cols_diff_check_skip = {
        'longitude',
        'dist_mi',
        'timezone',
        'elevation',
        'lbnl_convex_hull_existing_farms_2018',
        'lbnl_convex_hull_existing_farms_2021',
        'lcot',
        'trans_cap_cost',
        'electrolyzer_capex_per_mw',
        'capex_h2',
    }
    diff_ops = [DiffUnitOptions.ORIGINAL, DiffUnitOptions.PERCENTAGE]
    for col, opt in product(common_cols, diff_ops):
        col_name = f'{col}{opt}'
        assert col_name in difference

        if col in cols_diff_check_skip:
            continue

        mask = ~difference[col_name].isna()
        assert (difference.loc[mask, col_name] >= 0).all()
