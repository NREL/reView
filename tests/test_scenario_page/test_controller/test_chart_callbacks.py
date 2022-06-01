# -*- coding: utf-8 -*-
"""Chart callback tests."""
import pytest

from reView.pages.scenario.controller.callbacks import (
    options_chart_type,
    chart_tab_div_children,
)


def test_options_chart_type():
    """Test that correct option is shown if characterizations in config."""
    labels = {opt["label"] for opt in options_chart_type("Hydrogen")}
    assert "Characterizations" in labels

    labels = {opt["label"] for opt in options_chart_type("Hydrogen Minimal")}
    assert "Characterizations" not in labels


@pytest.mark.parametrize(
    "chart_choice, num_expected_tabs, should_be_missing",
    [
        ("cumsum", 4, None),
        ("histogram", 3, "X Variable"),
        ("box", 3, "X Variable"),
        ("scatter", 4, None),
        ("binned", 4, None),
        ("char_histogram", 3, "Additional Scenarios"),
    ],
)
def test_chart_tab_div_children(
    chart_choice, num_expected_tabs, should_be_missing
):
    """Test that correct tabs are shown for chart choice."""
    tabs = chart_tab_div_children(chart_choice)

    assert len(tabs) == num_expected_tabs
    if should_be_missing:
        # pylint: disable=no-member
        assert should_be_missing not in {t.label for t in tabs}
