# -*- coding: utf-8 -*-
"""The Navigation Bar layout.

Created on Sun Aug 23 14:59:00 2020

@author: travis
"""
import dash_bootstrap_components as dbc

from dash import dcc, html

from reView.layout.styles import BUTTON_STYLES


NAVBAR = html.Nav(
    id="top-level-navbar",
    style={
        "background-color": "#1663B5",
        "width": "99%",
        "height": "85px",
        "margin-left": "10px",
        "margin-right": "15px",
        "margin-top": "-10px",
        "margin-bottom": "15px",
        "border-radius": "5px",
        "position": "fixed",
        "box-shadow": (
            " 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 "
            "rgba(0, 0, 0, 0.19)"
        ),
        "zIndex": 9999,
        "font-family": "Times New Roman",
        "font-size": "48px",
        "font-face": "bold",
    },
    children=[
        html.Div(
            [
                html.H1(
                    "reView | ",
                    style={
                        "float": "left",
                        "position": "relative",
                        "color": "white",
                        "font-family": "Times New Roman",
                        "font-size": "48px",
                        "font-face": "bold",
                        "margin-bottom": 5,
                        "margin-left": 15,
                        "margin-top": 10,
                    },
                ),
                html.H2(
                    children=("  Renewable Energy Potential Projects"),
                    style={
                        "float": "left",
                        "position": "relative",
                        "color": "white",
                        "font-family": "Times New Roman",
                        "font-size": "28px",
                        "margin-bottom": 5,
                        "margin-left": 15,
                        "margin-top": 25,
                        "margin-right": 55,
                    },
                ),
            ]
        ),
        dcc.Link(
            dbc.Button(
                id="rev_link_button",
                children="reV Page",
                type="button",
                title=("Go to the reV viewer page."),
                size="lg",
                className="me-1",
                color="light",
                style=BUTTON_STYLES["navbar"]
            ),
            id="rev_link",
            href="/review/rev",
        ),
        dcc.Link(
            dbc.Button(
                id="reeds_link_button",
                children="ReEDS Page",
                className="me-1",
                size="lg",
                color="light",
                title=("Go to the ReEDS buildout viewer page."),
                style=BUTTON_STYLES["navbar"]
            ),
            id="reeds_link",
            href="/review/reeds",
        ),
        html.A(
            html.Img(
                src=("https://raw.githubusercontent.com/NREL/reView/"
                     "main/reView/static/nrel_logo.png"),
                className="twelve columns",
                style={
                    "height": "70px",
                    "width": "175px",
                    "float": "right",
                    "position": "relative",
                    "margin-left": "10px",
                    "margin-right": "10px",
                    "margin-top": "7px",
                    "border-bottom-right-radius": "3px",
                    "border-bottom-left-radius": "3px",
                    "border-top-left-radius": "3px",
                    "border-top-right-radius": "3px",
                    "border": "2px solid white"
                },
            ),
            href="https://www.nrel.gov/",
            target="_blank",
        ),
    ],
)

SIDE_BUTTON = html.Div(
    children=html.Button(
        "Additional Scenarios",
        id="open_side_options",
        style={
            "position": "fixed",
            "left": "-87px",
            "width": "21rem",
            "height": "4rem",
            "margin-top": "1000px",
            "background-color": "white",
            "border-radius": "6px 6px 0px 0px",
            "box-shadow": (
                " 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 "
                "rgba(0, 0, 0, 0.19)"
            ),
            "transform": "rotate(-270deg)"
        }
    )
)
