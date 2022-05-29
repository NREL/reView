# -*- coding: utf-8 -*-
"""Common reView callbacks. """
from dash.dependencies import Input, Output

from reView.app import app
from reView.layout.styles import RC_STYLES
from reView.utils import calls


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
    @calls.log
    def _toggle_reverse_color_button_style(click):
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

    return _toggle_reverse_color_button_style


def display_selected_tab_above_map(id_prefix):
    """Display the selected tab above the map.

    This method assumes you have all the elements added by
    `reView.components.divs.above_map_options_div` somewhere
    in your layout.

    Parameters
    ----------
    id_prefix : str
        A string representing the prefix of the button. This prefix
        should match the one used for the
        `reView.components.divs.above_map_options_div` function call.

    Returns
    -------
    callable
        A callable function used by dash. Users should NOT invoke this
        function themselves.
    """

    @app.callback(
        Output(f"{id_prefix}_state_options_div", "style"),
        Output(f"{id_prefix}_region_options_div", "style"),
        Output(f"{id_prefix}_basemap_options_div", "style"),
        Output(f"{id_prefix}_color_options_div", "style"),
        Input(f"{id_prefix}_options_tab", "value"),
    )
    @calls.log
    def _display_selected_tab_above_map(tab_choice):
        """Choose which map tabs to display."""
        # Styles
        styles = [{"display": "none"}] * 4
        order = ["state", "region", "basemap", "color"]
        idx = order.index(tab_choice)
        styles[idx] = {"width": "100%", "text-align": "center"}
        return styles[0], styles[1], styles[2], styles[3]

    return _display_selected_tab_above_map
