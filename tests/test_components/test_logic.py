# -*- coding: utf-8 -*-
"""Component logic tests."""
import pytest

from reView.components.logic import tab_styles


@pytest.mark.parametrize("tab_choice", ["a", "b", "c", "d", "e"])
def test_tab_styles_generic(tab_choice):
    """Test that only options for one chart tabs are displayed."""
    styles = tab_styles(tab_choice, options=["a", "b", "c", "d", "e"])
    assert sum(s != {"display": "none"} for s in styles) == 1


@pytest.mark.parametrize(
    "tab_choice", ["chart", "x_variable", "region", "scenarios"]
)
def test_tab_styles_chart(tab_choice):
    """Test that only options for one chart tabs are displayed."""
    styles = tab_styles(
        tab_choice, options=["chart", "x_variable", "region", "scenarios"]
    )
    assert sum(s != {"display": "none"} for s in styles) == 1


@pytest.mark.parametrize(
    "tab_choice", ["state", "region", "basemap", "color"]
)
def test_tab_styles_map(tab_choice):
    """Test that only options for one chart tabs are displayed."""
    styles = tab_styles(
        tab_choice, options=["state", "region", "basemap", "color"]
    )
    assert sum(s != {"display": "none"} for s in styles) == 1
