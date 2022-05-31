# -*- coding: utf-8 -*-
"""Component logic functions."""


def tab_styles(tab_choice, options):
    """Set correct tab styles for the chosen option."""
    styles = [{"display": "none"}] * len(options)
    idx = options.index(tab_choice)
    styles[idx] = {"width": "100%", "text-align": "center"}
    return styles
