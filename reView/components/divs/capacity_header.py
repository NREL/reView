# -*- coding: utf-8 -*-
"""A common capacity header."""
from dash import dcc, html


def capacity_header(id_prefix, style=None, class_name=None,
                    cap_title="Remaining Capacity",
                    count_title="Number of Sites", small=False):
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
    capacity_title : str, optional
        Title to place over remaining capacity printout.
    count_title : str, optional
        Title to place over remaining site count printout.
    small : boolen
        Reduce size of title and printouts.

    Returns
    -------
    dash.html
        A dash `html.Div` that displays the aggregate capacity and
        number of sites.
    """
    # Create ids
    cap_id = f"{id_prefix}_capacity_print"
    count_id = f"{id_prefix}_site_print"

    # Size options
    if small:
        capacity = html.H3(id=cap_id, children="")
        count = html.H3(id=count_id, children="")
    else:
        capacity = html.H1(id=cap_id, children="")
        count = html.H1(id=count_id, children="")

    # Print total capacity after all the filters are applied
    div = html.Div(
        style=style or {},
        children=[
            html.Div(
                [
                    html.H5(cap_title) if small else html.H2(cap_title),
                    dcc.Loading(capacity, type="circle"),
                ],
                className=class_name,
                style={"margin-bottom": "-10px"}
            ),

            # Print site count after all the filters are applied
            html.Div(
                [
                    html.H5(count_title) if small else html.H2(count_title),
                    dcc.Loading(count, type="circle"),
                ],
                className=class_name,
                style={"margin-bottom": "-10px", "margin-left": "150px"}
            )
        ]
    )

    return div
