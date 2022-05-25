# -*- coding: utf-8 -*-
"""Test layout option specifications."""
import reView.layout.options as opts


def _options():
    """Generator of options constants."""
    for const_name, const_val in vars(opts).items():
        if const_name.isupper() and const_name.endswith("OPTIONS"):
            yield const_val


def test_options_are_lists():
    """Test that options are lists."""

    for option in _options():
        assert isinstance(option, list)


def test_option_items_are_dicts_with_correct_keys():
    """Test that option items are dicts with labels and values."""

    for option in _options():
        for item in option:
            assert isinstance(item, dict)
            assert "label" in item
            assert "value" in item


def test_option_values_are_unique():
    """Test that values (not labels) are unique.

    This test is important because dropdowns "fail" is multiple items
    have the same "value". In particular, you cannot select the second
    item of two items that have the same "value", even if they have
    different "labels".
    """

    for option in _options():
        unique_values = {item["value"] for item in option}
        assert len(unique_values) == len(option)
