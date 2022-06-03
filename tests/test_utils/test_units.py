# -*- coding: utf-8 -*-
"""Tests for reView units."""
from reView import Q_


MINIMUM_VIABLE_UNITS = {
    "GW",
    "MW",
    "TW",
    "USD",
    "USD/MW",
    "USD/MWh",
    "USD/kW",
    "USD/kg",
    "category",
    "count",
    "degrees",
    "dollars",
    "index",
    "kg",
    "km",
    "kW",
    "m",
    "miles",
    "multiplier",
    "percent",
    "ratio",
    "square km",
    None,
}


def test_expected_units_work():
    """Test that all expected units can be initialized."""

    for unit in MINIMUM_VIABLE_UNITS:
        __ = Q_(1, unit)
