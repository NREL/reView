# -*- coding: utf-8 -*-
"""Common reView callbacks. """
from dash.dependencies import Input, Output

from reView.app import app
from reView.layout.styles import RC_STYLES
from reView.components.logic import tab_styles, format_capacity_title
from reView.utils import calls


# def download(id_prefix, figure_type):
#     """Download dataset from given figure.

#     Parameters
#     ----------
#     id_prefix : str
#         A string representing the prefix of the button. It is expected
#         that the id of the target button follows the format
#         "<id_prefix>_rev_color". "rev" and "reeds" currently available.
#     figure_type : str
#         A string representing the type of figure to download from. This can
#         be either "map" or "chart".

#     Returns
#     -------
#     callable
#         A callable function used by dash. Users should NOT invoke this
#         function themselves.
#     """
#     @app.callback(
#         Output(f"download", "data"),
#         Input(f"{id_prefix}_{figure_type}_download_button", "n_clicks")
#     )
#     @calls.log
#     def _download_data(n_clicks):
#         """Download data as CSV."""
    

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
        Output(f"{id_prefix}_map_rev_color", "children"),
        Output(f"{id_prefix}_map_rev_color", "style"),
        Input(f"{id_prefix}_map_rev_color", "n_clicks"),
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
        Output(f"{id_prefix}_map_state_options_div", "style"),
        Output(f"{id_prefix}_map_region_options_div", "style"),
        Output(f"{id_prefix}_map_basemap_options_div", "style"),
        Output(f"{id_prefix}_map_color_options_div", "style"),
        Input(f"{id_prefix}_map_options_tab", "value"),
    )
    @calls.log
    def _display_selected_tab_above_map(tab_choice):
        """Choose which map tabs to display."""
        return tab_styles(
            tab_choice, options=["state", "region", "basemap", "color"]
        )

    return _display_selected_tab_above_map


def capacity_print(id_prefix):
    """Update the aggregate capacity and number of sites values.

    This method assumes you have all the elements added by
    `reView.components.divs.capacity_header` somewhere
    in your layout.

    Parameters
    ----------
    id_prefix : str
        A string representing the prefix of the capacity headers.
        It is expected that the id of the divs follows the format
        "<id_prefix>_capacity_print" and "<id_prefix>__site_print".
        This is guaranteed if you use
        `reView.components.divs.capacity_header` to build your layout.

    Returns
    -------
    callable
        A callable function used by dash. Users should NOT invoke this
        function themselves.
    """

    @app.callback(
        Output(f"{id_prefix}_capacity_print", "children"),
        Output(f"{id_prefix}_site_print", "children"),
        Input(f"{id_prefix}_mapcap", "children"),
        Input(f"{id_prefix}_map", "selectedData"),
    )
    @calls.log
    def _capacity_print(map_capacity, map_selection):
        """Calculate total remaining capacity."""
        return format_capacity_title(map_capacity, map_selection)

    return _capacity_print
