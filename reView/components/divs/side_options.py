#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common side bar options.

Created on Mon Jul 10 11:42:35 2023

@author: twillia2
"""
import dash_bootstrap_components as dbc

from dash import dcc, html


def add_custom_props(component, **kwargs):
    prop_names = component._prop_names
    new_props = set(kwargs.keys()) - set(prop_names)
    if new_props:
        prop_names.extend(new_props)
    for k, v in kwargs.items():
        setattr(component, k, v)
    return component


SIDE_OPTIONS = html.Div(
    children=[
        dbc.Offcanvas(
            id="side_options",
            children=[
                html.H2(
                    children="Additional Scenarios",
                    title=("Add additional scenarios to the chart and "
                           "timeseries")
                ),
                html.Hr(style={"height": "1px"}),
                html.P(
                    children=(
                        "Select individual datasets to add to the chart and "
                        "timeseries graphs. Options will be prefiltered "
                        "by the dropdowns in the main options component."
                    )
                ),

                # Scenario grouping
                html.Div(
                    id="rev_additional_scenarios_div",
                    className="row",
                    children=[
                        # Submit Button to avoid repeated callbacks
                        html.Div(
                            className="twelve columns",
                            style={"margin-left": "50px"},
                            children=[
                                dbc.Button(
                                    id="rev_select_all_scenarios",
                                    children="ALL",
                                    color="white",
                                    n_clicks=0,
                                    size="lg",
                                    title="Click to select all available options",
                                    className="mb-1",
                                    style={
                                        "padding": "5px",
                                    }
                                ),
                                dbc.Button(
                                    id="rev_clear_all_scenarios",
                                    children="CLEAR",
                                    color="white",
                                    n_clicks=0,
                                    size="lg",
                                    title="Click to clear all selected options",
                                    className="mb-1",
                                    style={
                                        "padding": "5px",
                                    }
                                ),
                                dbc.Button(
                                    id="rev_submit_additional_scenarios",
                                    children="Submit",
                                    color="white",
                                    n_clicks=0,
                                    size="lg",
                                    title="Click to submit options",
                                    className="mb-1",
                                    style={
                                        "padding": "5px",
                                    }
                                )
                            ]
                        ),
                        html.Div(
                            className="eleven columns",
                            style={"margin-right": "-50px"},
                            children=add_custom_props(
                                dcc.Dropdown(
                                    id="rev_additional_scenarios",
                                    clearable=False,
                                    options=[
                                        {
                                            "label": "None",
                                            "value": "None",
                                        }
                                    ],
                                    multi=True,
                                ),
                            maxHeight=750
                            )
                        ),
                    ]
                )
            ],
            placement="start",
            is_open=False,
            style={
                "border-radius": "10px",
                "margin-left": "10px",
                "margin-top": "92px",
                "font-face": "bold",
                "font-family": "Times New Roman",
                "font-size": "24px",
                "width": "700px"
            }
        )
    ]
)
