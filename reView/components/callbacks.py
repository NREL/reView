# -*- coding: utf-8 -*-
"""Common reView callbacks. """
from dash.dependencies import Input, Output

from reView.app import app
from reView.layout.styles import RC_STYLES


def toggle_reverse_color_button_style(id_prefix):
    """Change the style of the "reverse color" button when clicked.

    This method assumes you have a `html.Button` in your layout with an
    id of "<id_prefix>_rev_color".

    Parameters
    ----------
    id_prefix : str
        A string representing the prefix of the button. It is expected
        that the id of the target button follows the format
        "<id_prefix>_rev_color".

    Returns
    -------
    callable
        A callable function used by dash. Users should NOT invoke this
        function themselves.
    """

    @app.callback(
        Output(f"{id_prefix}_rev_color", "children"),
        Output(f"{id_prefix}_rev_color", "style"),
        Input(f"{id_prefix}_rev_color", "n_clicks"),
    )
    def _toggle_reverse_color_button(click):
        """Toggle Reverse Color on/off."""
        if not click:
            click = 0
        if click % 2 == 1:
            children = "Reverse: Off"
            style = RC_STYLES["off"]
        else:
            children = "Reverse: On"
            style = RC_STYLES["on"]

        return children, style

    return _toggle_reverse_color_button
