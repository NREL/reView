# -*- coding: utf-8 -*-
"""A common capacity header."""
from dash import dcc, html


def capacity_header(id_prefix, style=None, class_name=None):
    """Standard capacity output header divs.

    Parameters
    ----------
    id_prefix : str
        A string representing the prefix to use for individual
        components. Assuming `id_prefix="map"`, the components are:
            - map_options_tab
                A `dcc.Tabs` with tabs representing the map options for
                user.
            - map_state_options_div
                A `html.Div` that holds the `dcc.Dropdown` that includes
                all state options.
    style : dict
        A dictionary containing html style components.
    class_name : str, optional
        The className of the capacity header divs.
        By default, `None`.

    Returns
    -------
    dash.html
        A dash `html.Div` that displays the aggregate capacity and
        number of sites.
    """
    # Print total capacity after all the filters are applied
    return html.Div(
        children=[
            html.Div(
                [
                    html.H5("Remaining Generation Capacity: "),
                    dcc.Loading(
                        children=[
                            html.H1(
                                id=f"{id_prefix}_capacity_print", children=""
                            ),
                        ],
                        type="circle",
                    ),
                ],
                className=class_name,
            ),
            # Print total capacity after all the filters are applied
            html.Div(
                [
                    html.H5("Number of Sites: "),
                    dcc.Loading(
                        children=[
                            html.H1(id=f"{id_prefix}_site_print", children=""),
                        ],
                        type="circle",
                    ),
                ],
                className=class_name,
            ),
        ],
        style=style or {},
    )
