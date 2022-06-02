"""View reV results using a configuration file.

Things to do:
    - Move styling to CSS
    - Improve caching
    - Speed up everything
    - Download option
    - Automate startup elements
    - Build categorical variable charts
"""
import hashlib
import json
import logging
import os

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from reView.app import app
from reView.layout.styles import BUTTON_STYLES, TABLET_STYLE
from reView.layout.options import (
    CHART_OPTIONS,
    COLOR_OPTIONS,
    COLOR_Q_OPTIONS,
)
from reView.components.callbacks import (
    capacity_print,
    toggle_reverse_color_button_style,
    display_selected_tab_above_map,
)
from reView.components.logic import tab_styles
from reView.components.map import Map, build_title
from reView.pages.scenario.controller.element_builders import Plots
from reView.pages.scenario.controller.selection import (
    all_files_from_selection,
    choose_scenario,
    parse_selection,
    scrape_variable_options,
)
from reView.pages.scenario.model import (
    apply_all_selections,
    calc_least_cost,
    cache_map_data,
    cache_table,
    cache_chart_tables,
)
from reView.utils.bespoke import BespokeUnpacker
from reView.utils.constants import SKIP_VARS
from reView.utils.functions import convert_to_title, callback_trigger
from reView.utils.config import Config
from reView.utils import calls

logger = logging.getLogger(__name__)
COMMON_CALLBACKS = [
    capacity_print(id_prefix="rev"),
    toggle_reverse_color_button_style(id_prefix="rev"),
    display_selected_tab_above_map(id_prefix="rev"),
]


def build_specs(scenario, project):
    """Calculate the percentage of each scenario present."""
    config = Config(project)
    specs = config.parameters
    dct = specs[scenario]
    table = """| Variable | Value |\n|----------|------------|\n"""
    for variable, value in dct.items():
        row = "| {} | {} |\n".format(variable, value)
        table = table + row
    return table


def build_spec_split(path, project):
    """Calculate the percentage of each scenario present."""
    df = cache_table(project, path)
    scenarios, counts = np.unique(df["scenario"], return_counts=True)
    total = df.shape[0]
    percentages = [counts[i] / total for i in range(len(counts))]
    percentages = [round(p * 100, 4) for p in percentages]
    pdf = pd.DataFrame(dict(p=percentages, s=scenarios))
    pdf = pdf.sort_values("p", ascending=False)
    table = """| Scenario | Percentage |\n|----------|------------|\n"""
    for _, row in pdf.iterrows():
        row = "| {} | {}% |\n".format(row["s"], row["p"])
        table = table + row
    return table


def chart_tab_div_children(chart_choice):
    """Choose which chart tabs to display based on the option."""

    children = [
        dcc.Tab(
            value="chart",
            label="Chart Type",
            style=TABLET_STYLE,
            selected_style=TABLET_STYLE,
        )
    ]

    # Add x variable option if needed
    if chart_choice not in {"box", "histogram"}:
        children += [
            dcc.Tab(
                value="x_variable",
                label="X Variable",
                style=TABLET_STYLE,
                selected_style=TABLET_STYLE,
            )
        ]

    children += [
        dcc.Tab(
            value="region",
            label="Region",
            style=TABLET_STYLE,
            selected_style=TABLET_STYLE,
        )
    ]

    if chart_choice not in {"char_histogram"}:
        children += [
            dcc.Tab(
                value="scenarios",
                label="Additional Scenarios",
                style=TABLET_STYLE,
                selected_style=TABLET_STYLE,
            )
        ]

    return children


@calls.log
def options_chart_type(project):
    """Add characterization plot option, if necessary."""
    if Config(project).characterizations_cols:
        return CHART_OPTIONS

    return CHART_OPTIONS[:-1]


def scenario_dropdowns(groups, class_names=None):
    """Return list of dropdown options for a project's file selection."""
    dropdowns = []
    class_names = class_names or ["six columns"]
    colors = ["#b4c9e0", "#e3effc"]

    for ind, (group, options) in enumerate(groups.items()):
        color = colors[ind % 2]

        dropdown = html.Div(
            [
                html.Div(
                    [html.P(group)],
                    className=class_names[0],
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            options=options,
                            value=options[0]["value"],
                            optionHeight=75,
                        )
                    ],
                    className=class_names[-1],
                ),
            ],
            className="row",
            style={"background-color": color},
        )

        dropdowns.append(dropdown)

    drop_div = html.Div(
        children=dropdowns,
        style={"border": "4px solid #1663b5", "padding": "2px"},
    )

    return drop_div


@app.callback(
    Output("recalculate_with_new_costs", "hidden"),
    Input("project", "value"),
    Input("toggle_options", "n_clicks"),
)
def disable_recalculate_with_new_costs(project, __):
    """Disable recalculate option based on config."""
    return not Config(project).parameters


@app.callback(
    Output("map_function_div", "hidden"),
    Output("map_function", "value"),
    Input("project", "value"),
    Input("toggle_options", "n_clicks"),
)
def disable_mapping_function_dev(project, __):
    """Disable mapping option based on config."""
    return Config(project).demand_data is None, "None"


@app.callback(
    Output("chart_options", "options"),
    Input("project", "value"),
)
@calls.log
def dropdown_chart_types(project):
    """Add characterization plot option, if necessary."""
    return options_chart_type(project)


@app.callback(
    Output("rev_map_color_options", "options"),
    Output("rev_map_color_options", "value"),
    Input("submit", "n_clicks"),
    State("variable", "value"),
    State("project", "value"),
    State("map_signal", "children"),
    State("rev_map_color_options", "value"),
)
@calls.log
def dropdown_colors(submit, variable, project, signal, old_value):
    """Provide qualitative color options for categorical data."""

    # To figure out if we need to update we need these
    if not signal:
        raise PreventUpdate  # @IgnoreException
    old_variable = json.loads(signal)["y"]
    config = Config(project)
    units = config.units.get(variable)
    old_units = config.units.get(old_variable)

    # There is only one condition where we have to do this
    if old_variable == variable:
        raise PreventUpdate  # @IgnoreException
    if old_units == units:
        raise PreventUpdate  # @IgnoreException
    if old_units != "category" and units != "category":
        raise PreventUpdate  # @IgnoreException

    # Now return the appropriate options
    if units == "category":
        options = COLOR_Q_OPTIONS
        value = "T10"
    else:
        options = COLOR_OPTIONS
        value = "Viridis"

    return options, value


# pylint: disable=no-member
@app.callback(
    Output("minimizing_scenario_options", "children"),
    Input("url", "pathname"),
    Input("project", "value"),
    Input("minimizing_variable", "value"),
    State("submit", "n_clicks"),
)
@calls.log
def dropdown_minimizing_scenarios(url, project, minimizing_variable, n_clicks):
    """Update the options given a project."""

    logger.debug("URL: %s", url)

    # We need the project configuration
    config = Config(project)

    if config.options is not None:
        groups = {}

        for col in config.options.columns:
            if col in {"name", "file"} or col == minimizing_variable:
                continue
            # pylint: disable=unsubscriptable-object
            options = config.options[col].unique()
            dropdown_options = []
            for op in options:
                try:
                    label = "{:,}".format(op)
                except ValueError:
                    label = str(op)
                ops = {"label": label, "value": op}
                dropdown_options.append(ops)
            groups[col] = dropdown_options

        return scenario_dropdowns(groups)

    # Find the files
    scenario_outputs_path = config.directory / ".review"
    scenario_outputs = [
        str(f) for f in scenario_outputs_path.glob("least*.csv")
    ]
    scenario_outputs = []
    scenario_originals = [str(file) for file in config.files.values()]
    files = scenario_originals + scenario_outputs
    names = [os.path.basename(f).replace("_sc.csv", "") for f in files]
    names = [convert_to_title(name) for name in names]
    file_list = dict(zip(names, files))

    scenario_options = [
        {"label": key, "value": os.path.expanduser(file)}
        for key, file in file_list.items()
    ]

    if not scenario_options:
        scenario_options = [{"label": "None", "value": None}]

    return scenario_dropdowns(
        {"Scenario": scenario_options},
        class_names=["four columns", "eight columns"],
    )


@app.callback(
    Output("minimizing_target", "options"),
    Output("minimizing_target", "value"),
    Input("minimizing_scenario_options", "children"),
    State("project", "value"),
)
def dropdown_minimizing_targets(scenario_options, project):
    """Set the minimizing target options."""
    logger.debug("Setting minimizing target options")
    config = Config(project)
    path = choose_scenario(scenario_options, config)
    target_options = []
    if path and os.path.exists(path):
        data = pd.read_csv(path)
        columns = [c for c in data.columns if c.lower() not in SKIP_VARS]
        titles = {col: convert_to_title(col) for col in columns}
        titles.update(config.titles)
        if titles:
            for k, v in titles.items():
                target_options.append({"label": v, "value": k})

    if not target_options:
        target_options = [{"label": "None", "value": "None"}]

    return target_options, target_options[-1]["value"]


@app.callback(
    Output("project", "options"),
    Output("project", "value"),
    Input("url", "pathname"),
    State("submit", "n_clicks"),
)
@calls.log
def dropdown_projects(pathname, n_clicks):
    """Update project options."""

    # Open config json
    project_options = [
        {"label": project, "value": project}
        # pylint: disable=not-an-iterable
        for project in Config.sorted_projects
    ]
    return project_options, project_options[0]["value"]


@app.callback(
    Output("minimizing_plot_value", "options"),
    Input("minimizing_scenario_options", "children"),
    State("project", "value"),
)
def dropdown_minimizing_plot_options(scenario_options, project):
    """Set the minimizing plot options."""
    logger.debug("Setting minimizing plot options")
    config = Config(project)
    path = choose_scenario(scenario_options, config)
    plot_options = [{"label": "Variable", "value": "Variable"}]
    if path and os.path.exists(path):
        data = pd.read_csv(path)
        columns = [c for c in data.columns if c.lower() not in SKIP_VARS]
        titles = {col: convert_to_title(col) for col in columns}
        titles.update(config.titles)
        if titles:
            for k, v in titles.items():
                plot_options.append({"label": v, "value": k})

    return plot_options


@app.callback(
    Output("scenario_a_options", "children"),
    Output("scenario_b_options", "children"),
    Input("url", "pathname"),
    Input("project", "value"),
    State("submit", "n_clicks"),
)
@calls.log
def dropdown_scenarios(url, project, n_clicks):
    """Update the options given a project."""
    logger.debug("URL: %s", url)

    # We need the project configuration
    config = Config(project)

    if config.options is not None:
        groups = {}

        for col in config.options.columns:
            if col in {"name", "file"}:
                continue
            # pylint: disable=unsubscriptable-object
            options = config.options[col].unique()
            dropdown_options = []
            for op in options:
                try:
                    label = "{:,}".format(op)
                except ValueError:
                    label = str(op)
                ops = {"label": label, "value": op}
                dropdown_options.append(ops)
            groups[col] = dropdown_options

        return (
            scenario_dropdowns(groups),
            scenario_dropdowns(groups),
        )

    # Find the files
    scenario_outputs_path = config.directory.joinpath(".review")
    scenario_outputs = [
        str(f) for f in scenario_outputs_path.glob("least*.csv")
    ]
    scenario_originals = [str(file) for file in config.files.values()]
    scenario_originals.sort()
    files = scenario_originals + scenario_outputs
    names = [os.path.basename(f).replace("_sc.csv", "") for f in files]
    names = [convert_to_title(name) for name in names]
    file_list = dict(zip(names, files))

    scenario_options = [
        {"label": key, "value": os.path.expanduser(file)}
        for key, file in file_list.items()
    ]

    if not scenario_options:
        scenario_options = [{"label": "None", "value": None}]

    return (
        scenario_dropdowns(
            {"Scenario": scenario_options},
            class_names=["four columns", "eight columns"],
        ),
        scenario_dropdowns(
            {"Scenario": scenario_options},
            class_names=["four columns", "eight columns"],
        ),
    )


@app.callback(
    Output("variable", "options"),
    Output("variable", "value"),
    Output("filter_variables_1", "options"),
    Output("filter_variables_2", "options"),
    Output("filter_variables_3", "options"),
    Output("filter_variables_4", "options"),
    Input("url", "href"),
    Input("scenario_a_options", "children"),
    Input("scenario_b_options", "children"),
    Input("scenario_b_div", "style"),
    Input("project", "value"),
)
@calls.log
def dropdown_variables(
    url, scenario_a_options, scenario_b_options, b_div, project
):
    """Update variable dropdown options."""

    variable_options = scrape_variable_options(
        project, scenario_a_options, scenario_b_options, b_div
    )

    if not variable_options:
        print("NO VARIABLE OPTIONS FOUND!")
        variable_options = [{"label": "None", "value": "None"}]

    return (
        variable_options,
        "capacity",
        variable_options,
        variable_options,
        variable_options,
        variable_options,
    )


@app.callback(
    Output("chart_x_var_options", "options"),
    Output("chart_x_var_options", "value"),
    Input("scenario_a_options", "children"),
    Input("scenario_b_options", "children"),
    Input("scenario_b_div", "style"),
    Input("chart_options", "value"),
    State("project", "value"),
)
def dropdown_x_variables(
    scenario_a_options, scenario_b_options, b_div, chart_type, project
):
    logger.debug("Setting X variable options")
    if chart_type == "char_histogram":
        config = Config(project)
        variable_options = [
            {"label": config.titles.get(x, convert_to_title(x)), "value": x}
            for x in config.characterizations_cols
        ]
        val = variable_options[0]["value"]
    else:
        variable_options = scrape_variable_options(
            project, scenario_a_options, scenario_b_options, b_div
        )
        val = "capacity"

    if not variable_options:
        variable_options = [{"label": "None", "value": "None"}]
        val = "None"

    return variable_options, val


@app.callback(
    Output("additional_scenarios", "options"),
    Input("url", "pathname"),
    Input("project", "value"),
    State("submit", "n_clicks"),
)
@calls.log
def dropdowns_additional_scenarios(
    url,
    project,
    n_clicks,
):
    """Update the additional scenarios options given a project."""
    logger.debug("URL: %s", url)

    # We need the project configuration
    config = Config(project)

    # Find the files
    scenario_outputs_path = config.directory / ".review"
    scenario_outputs = [
        str(f) for f in scenario_outputs_path.glob("least*.csv")
    ]
    scenario_outputs = []
    scenario_originals = [str(file) for file in config.files.values()]
    files = scenario_originals + scenario_outputs
    names = [os.path.basename(f).replace("_sc.csv", "") for f in files]
    names = [
        " ".join([n.capitalize() for n in name.split("_")]) for name in names
    ]
    file_list = dict(zip(names, files))

    scenario_options = [
        {"label": key, "value": os.path.expanduser(file)}
        for key, file in file_list.items()
    ]

    if not scenario_options:
        scenario_options = [{"label": "None", "value": None}]

    least_cost_options = []
    for key, file in file_list.items():
        if file in config.files.values():
            option = {"label": key, "value": str(file)}
            least_cost_options.append(option)

    return scenario_options


@app.callback(
    Output("chart", "figure"),
    Output("chart_loading", "style"),
    Input("map_signal", "children"),
    Input("chart_options", "value"),
    Input("rev_map", "selectedData"),
    Input("chart_point_size", "value"),
    Input("chosen_map_options", "children"),
    Input("chart_region", "value"),
    Input("rev_map_color_min", "value"),
    Input("rev_map_color_max", "value"),
    Input("chart_x_bin", "value"),
    Input("chart_alpha", "value"),
    State("chart", "selectedData"),
    State("project", "value"),
    State("chart", "relayoutData"),
    State("map_function", "value"),
)
@calls.log
def figure_chart(
    signal,
    chart,
    map_selection,
    point_size,
    op_values,
    region,
    user_ymin,
    user_ymax,
    bin_size,
    alpha,
    chart_selection,
    project,
    chart_view,
    map_func,
):
    """Make one of a variety of charts."""

    # Unpack the signal
    signal_dict = json.loads(signal)
    x = signal_dict["x"]
    y = signal_dict["y"]
    project = signal_dict["project"]

    if (
        chart == "char_histogram"
        and x not in Config(project).characterizations_cols
    ):
        raise PreventUpdate  # @IgnoreException

    # Get the data frames
    dfs = cache_chart_tables(signal_dict, region)
    if all(df.empty for df in dfs.values()):
        figure = go.Figure()
        figure.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "No matching data found",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 28},
                }
            ],
        )
        return figure

    # Turn the map selection object into indices
    if map_selection:
        dfs = {
            k: apply_all_selections(
                df,
                map_func,
                project,
                chart_selection,
                map_selection,
                clicksel=None,
            )[0]
            for k, df in dfs.items()
        }

    if chart_selection:
        n_points_selected = len(chart_selection["points"])
        title = f"Selected point count: {n_points_selected:,}"
    else:
        title = None

    plotter = Plots(
        project,
        dfs,
        plot_title=title,
        point_size=point_size,
        user_scale=(user_ymin, user_ymax),
        alpha=alpha,
    )

    if chart == "cumsum":
        fig = plotter.cumulative_sum(x, y)
    elif chart == "scatter":
        fig = plotter.scatter(x, y)
    elif chart == "binned":
        fig = plotter.binned(x, y, bin_size=bin_size)
    elif chart == "histogram":
        fig = plotter.histogram(y)
    elif chart == "char_histogram":
        fig = plotter.char_hist(x)
    elif chart == "box":
        fig = plotter.box(y)

    return fig, {"float": "right"}


@app.callback(
    Output("rev_map", "figure"),
    Output("rev_mapcap", "children"),
    Output("rev_map", "clickData"),
    Output("map_loading", "style"),
    Input("map_signal", "children"),
    Input("rev_map_basemap_options", "value"),
    Input("rev_map_color_options", "value"),
    Input("chart", "selectedData"),
    Input("rev_map_point_size", "value"),
    Input("rev_map_rev_color", "n_clicks"),
    Input("rev_map_color_min", "value"),
    Input("rev_map_color_max", "value"),
    Input("rev_map", "selectedData"),
    Input("rev_map", "clickData"),
    State("project", "value"),
    State("map_function", "value"),
)
@calls.log
def figure_map(
    signal,
    basemap,
    color,
    chart_selection,
    point_size,
    reverse_color_clicks,
    color_ymin,
    color_ymax,
    map_selection,
    click_selection,
    project,
    map_function,
):
    """Make the scatter plot map."""

    signal_dict = json.loads(signal)
    df = cache_map_data(signal_dict)

    df, demand_data = apply_all_selections(
        df,
        map_function,
        project,
        chart_selection,
        map_selection,
        click_selection,
    )

    if "clickData" in callback_trigger() and "turbine_y_coords" in df:
        unpacker = BespokeUnpacker(df, click_selection)
        df = unpacker.unpack_turbines()

    # Use demand counts if available
    if "demand_connect_count" in df:
        color_var = "demand_connect_count"
    else:
        color_var = signal_dict["y"]

    title = build_title(df, color_var, project, map_selection=map_selection)

    # Build figure
    map_builder = Map(
        df=df,
        color_var=color_var,
        plot_title=title,
        project=project,
        basemap=basemap,
        colorscale=color,
        color_min=color_ymin,
        color_max=color_ymax,
        demand_data=demand_data,
    )
    figure = map_builder.figure(
        point_size=point_size,
        reverse_color=reverse_color_clicks % 2 == 1,
    )
    mapcap = df[["sc_point_gid", "print_capacity"]].to_dict()

    return figure, json.dumps(mapcap), None, {"float": "left"}


# @app.callback(
#     Output("recalc_a_options", "children"),
#     [
#         Input("project", "value"),
#         Input("scenario_a", "value"),
#     ],
#     [
#         State("recalc_table", "children"),
#     ],
# )
# @calls.log
# def options_recalc_a(project, scenario, recalc_table):
#     """Update the drop down options for each scenario."""

#     data = Data(project)
#     recalc_table = json.loads(recalc_table)
#     scenario = os.path.basename(scenario).replace("_sc.csv", "")
#     if scenario not in data.scenarios:
#         raise PreventUpdate

#     if not data.parameters:
#         raise PreventUpdate

#     table = recalc_table["scenario_a"]
#     original_table = data.original_parameters(scenario)
#     children = [
#         # FCR A
#         html.Div(
#             [
#                 html.P(
#                     "FCR % (A): ",
#                     className="three columns",
#                     style={"height": "60%"},
#                 ),
#                 dcc.Input(
#                     id="fcr1",
#                     type="number",
#                     className="nine columns",
#                     style={"height": "60%"},
#                     value=table["fcr"],
#                     placeholder=original_table["fcr"],
#                 ),
#             ],
#             className="row",
#         ),
#         # CAPEX A
#         html.Div(
#             [
#                 html.P(
#                     "CAPEX $/KW (A): ",
#                     className="three columns",
#                     style={"height": "60%"},
#                 ),
#                 dcc.Input(
#                     id="capex1",
#                     type="number",
#                     className="nine columns",
#                     style={"height": "60%"},
#                     value=table["capex"],
#                     placeholder=original_table["capex"],
#                 ),
#             ],
#             className="row",
#         ),
#         # OPEX A
#         html.Div(
#             [
#                 html.P(
#                     "OPEX $/KW (A): ",
#                     className="three columns",
#                     style={"height": "60%"},
#                 ),
#                 dcc.Input(
#                     id="opex1",
#                     type="number",
#                     className="nine columns",
#                     style={"height": "60%"},
#                     value=table["opex"],
#                     placeholder=original_table["opex"],
#                 ),
#             ],
#             className="row",
#         ),
#         # Losses A
#         html.Div(
#             [
#                 html.P(
#                     "Losses % (A): ",
#                     className="three columns",
#                     style={"height": "60%"},
#                 ),
#                 dcc.Input(
#                     id="losses1",
#                     type="number",
#                     className="nine columns",
#                     value=table["losses"],
#                     placeholder=original_table["losses"],
#                     style={"height": "60%"},
#                 ),
#             ],
#             className="row",
#         ),
#     ]

#     return children


# @app.callback(
#     Output("recalc_b_options", "children"),
#     [
#         Input("project", "value"),
#         Input("scenario_b", "value"),
#     ],
#     [
#         State("recalc_table", "children"),
#     ],
# )
# @calls.log
# def options_recalc_b(project, scenario, recalc_table):
#     """Update the drop down options for each scenario."""
#     data = Data(project)
#     recalc_table = json.loads(recalc_table)
#     if scenario not in data.scenarios:
#         raise PreventUpdate

#     if not data.parameters:
#         raise PreventUpdate

#     scenario = os.path.basename(scenario).replace("_sc.csv", "")
#     table = recalc_table["scenario_b"]
#     original_table = data.original_parameters(scenario)
#     scenario = os.path.basename(scenario).replace("_sc.csv", "")
#     table = recalc_table["scenario_b"]
#     original_table = data.original_parameters(scenario)
#     children = [
#         # FCR B
#         html.Div(
#             [
#                 html.P(
#                     "FCR % (B): ",
#                     className="three columns",
#                     style={"height": "60%"},
#                 ),
#                 dcc.Input(
#                     id="fcr2",
#                     type="number",
#                     className="nine columns",
#                     style={"height": "60%"},
#                     value=table["fcr"],
#                     placeholder=original_table["fcr"],
#                 ),
#             ],
#             className="row",
#         ),
#         # CAPEX B
#         html.Div(
#             [
#                 html.P(
#                     "CAPEX $/KW (B): ",
#                     className="three columns",
#                     style={"height": "60%"},
#                 ),
#                 dcc.Input(
#                     id="capex2",
#                     type="number",
#                     className="nine columns",
#                     style={"height": "60%"},
#                     value=table["capex"],
#                     placeholder=original_table["capex"],
#                 ),
#             ],
#             className="row",
#         ),
#         # OPEX B
#         html.Div(
#             [
#                 html.P(
#                     "OPEX $/KW (B): ",
#                     className="three columns",
#                     style={"height": "60%"},
#                 ),
#                 dcc.Input(
#                     id="opex2",
#                     type="number",
#                     className="nine columns",
#                     style={"height": "60%"},
#                     value=table["opex"],
#                     placeholder=original_table["opex"],
#                 ),
#             ],
#             className="row",
#         ),
#         # Losses B
#         html.Div(
#             [
#                 html.P(
#                     "Losses % (B): ",
#                     className="three columns",
#                     style={"height": "60%"},
#                 ),
#                 dcc.Input(
#                     id="losses2",
#                     type="number",
#                     className="nine columns",
#                     value=table["losses"],
#                     placeholder=original_table["losses"],
#                     style={"height": "60%"},
#                 ),
#             ],
#             className="row",
#         ),
#     ]

#     return children


@app.callback(
    Output("chart_data_signal", "children"),
    Input("variable", "value"),
    Input("chart_x_var_options", "value"),
    Input("rev_map_state_options", "value"),
)
@calls.log
def retrieve_chart_tables(y, x, state):
    """Store the signal used to get the set of tables needed for the chart."""
    signal = json.dumps([y, x, state])
    return signal


@app.callback(
    Output("filter_store", "children"),
    Input("submit", "n_clicks"),
    State("filter_variables_1", "value"),
    State("filter_variables_2", "value"),
    State("filter_variables_3", "value"),
    State("filter_variables_4", "value"),
    State("filter_1", "value"),
    State("filter_2", "value"),
    State("filter_3", "value"),
    State("filter_4", "value"),
)
@calls.log
def retrieve_filters(submit, var1, var2, var3, var4, q1, q2, q3, q4):
    """Retrieve filter variable names and queries."""

    variables = [var1, var2, var3, var4]
    queries = [q1, q2, q3, q4]

    filters = []
    for i, var in enumerate(variables):
        if var and queries[i]:
            filters.append(" ".join([var, queries[i]]))

    return json.dumps(filters)


@app.callback(
    Output("map_signal", "children"),
    Output("pca_plot_1", "clickData"),
    Output("pca_plot_2", "clickData"),
    Input("submit", "n_clicks"),
    Input("rev_map_state_options", "value"),
    Input("rev_map_region_options", "value"),
    Input("chart_options", "value"),
    Input("chart_x_var_options", "value"),
    Input("additional_scenarios", "value"),
    Input("filter_store", "children"),
    Input("pca_plot_1", "clickData"),
    Input("pca_plot_2", "clickData"),
    State("project", "value"),
    State("variable", "value"),
    State("difference", "value"),
    State("mask", "value"),
    State("recalc_table", "children"),
    State("recalc_tab", "value"),
    State("difference_units", "value"),
    State("scenario_a_options", "children"),
    State("scenario_b_options", "children"),
    State("minimizing_scenarios", "style"),
    State("minimizing_scenario_options", "children"),
    State("minimizing_variable", "value"),
    State("minimizing_target", "value"),
    State("minimizing_plot_value", "value"),
    State("pca_plot_map_value", "value"),
    State("pca_plot_region", "value"),
)
@calls.log
def retrieve_signal(
    submit,
    states,
    regions,
    chart,
    x,
    scenarios,
    filter_store,
    pca1_click_selection,
    pca2_click_selection,
    project,
    y,
    diff,
    mask,
    recalc_table,
    recalc,
    diff_units,
    scenario_a_options,
    scenario_b_options,
    minimizing_scenarios_style,
    minimizing_scenario_options,
    minimizing_variable,
    minimizing_target,
    minimizing_plot_value,
    pca_plot_value,
    pca_plot_region,
):
    """Create signal for sharing data between map and chart with dependence."""
    trig = callback_trigger()

    # Get/build the value scale table
    config = Config(project)

    # Unpack recalc table
    if recalc_table:
        recalc_table = json.loads(recalc_table)

    lowest_scenario_open = (
        minimizing_scenarios_style
        and minimizing_scenarios_style.get("display") != "none"
    )

    if lowest_scenario_open:
        if minimizing_variable in config.low_cost_groups:
            paths = [
                config.directory / file
                for file in config.low_cost_groups[minimizing_variable]
            ]
        else:
            options = parse_selection(minimizing_scenario_options)
            df = all_files_from_selection(options, config)
            if "file" in df:
                paths = [Path(p) for p in df["file"].values]
            else:
                paths = [config.files.get(f"{name}") for name in df["name"]]

        tag = hashlib.sha1(str.encode(str(paths))).hexdigest()
        fname = (
            f"least_{minimizing_target}_by_{minimizing_variable}_{tag}_sc.csv"
        )
        # Build full paths and create the target file
        lc_path = config.directory / ".review" / fname
        lc_path.parent.mkdir(parents=True, exist_ok=True)
        # calculator = LeastCost(project)
        calc_least_cost(paths, lc_path, by=minimizing_target)
        if minimizing_plot_value == "Variable":
            y = "scenario"
        else:
            y = minimizing_plot_value
        signal = {
            "filters": [],
            "mask": "off",
            "path": str(lc_path),
            "path2": None,
            "project": project,
            "recalc": recalc,
            "recalc_table": recalc_table,
            "added_scenarios": [],
            "regions": regions,
            "states": states,
            "x": x,
            "y": y,
        }
        return json.dumps(signal), None, None

    # Prevent the first trigger when difference is off
    if "scenario_b" in trig and diff == "off":
        raise PreventUpdate

    # Prevent the first trigger when mask is off
    if "mask" in trig and mask == "off":
        raise PreventUpdate

    if pca1_click_selection and pca1_click_selection.get("points"):
        path = pca1_click_selection["points"][0]["customdata"][0]
        path2 = None
        y = pca_plot_value
        states = [] if pca_plot_region == "CONUS" else [pca_plot_region]
    elif pca2_click_selection and pca2_click_selection.get("points"):
        path = pca2_click_selection["points"][0]["customdata"][0]
        path2 = None
        y = pca_plot_value
        states = [] if pca_plot_region == "CONUS" else [pca_plot_region]
    else:
        path = choose_scenario(scenario_a_options, config)
        path = os.path.expanduser(path)
        if diff == "off" and mask == "off":
            path2 = None
        else:
            path2 = choose_scenario(scenario_b_options, config)
            path2 = os.path.expanduser(path2)

    logger.debug("path = %s", path)
    logger.debug("path2 = %s", path2)

    if scenarios:
        scenarios = [os.path.expanduser(path) for path in scenarios]

    # Pack up the filters
    if filter_store:
        filters = json.loads(filter_store)
    else:
        filters = []

    if diff == "on":
        y = f"{y}{diff_units}"

    # Let's just recycle all this for the chart
    signal = {
        "filters": filters,
        "mask": mask,
        "path": str(path) if path else path,
        "path2": str(path2) if path2 else path2,
        "project": project,
        "recalc": recalc,
        "recalc_table": recalc_table,
        "added_scenarios": scenarios,
        "regions": regions,
        "states": states,
        "x": x,
        "y": y,
    }
    return json.dumps(signal), None, None


@app.callback(
    Output("recalc_table", "children"),
    Input("fcr1", "value"),
    Input("capex1", "value"),
    Input("opex1", "value"),
    Input("losses1", "value"),
    Input("fcr2", "value"),
    Input("capex2", "value"),
    Input("opex2", "value"),
    Input("losses2", "value"),
    Input("project", "value"),
)
def retrieve_recalc_parameters(
    fcr1, capex1, opex1, losses1, fcr2, capex2, opex2, losses2, project
):
    """Retrieve all given recalc values and store them."""
    trig = callback_trigger()
    if "project" in trig:
        recalc_table = {
            "scenario_a": {
                "fcr": None,
                "capex": None,
                "opex": None,
                "losses": None,
            },
            "scenario_b": {
                "fcr": None,
                "capex": None,
                "opex": None,
                "losses": None,
            },
        }
    else:
        recalc_table = {
            "scenario_a": {
                "fcr": fcr1,
                "capex": capex1,
                "opex": opex1,
                "losses": losses1,
            },
            "scenario_b": {
                "fcr": fcr2,
                "capex": capex2,
                "opex": opex2,
                "losses": losses2,
            },
        }
    return json.dumps(recalc_table)


@app.callback(
    Output("minimizing_variable", "options"),
    Input("project", "value"),
    Input("minimizing_scenarios", "style"),
)
def set_minimizing_variable_options(project, minimizing_scenarios_style):
    """Set the minimizing variable options."""
    logger.debug("Setting variable target options")
    config = Config(project)
    variable_options = [{"label": "None", "value": "None"}]
    is_showing = (
        minimizing_scenarios_style
        and minimizing_scenarios_style.get("display") != "none"
    )
    if not is_showing:
        return variable_options

    if config.options is not None:
        variable_options += [
            {"label": col, "value": col}
            for col in config.options.columns
            if col not in {"name", "file"}
        ]
    low_cost_group_options = [
        {"label": g, "value": g} for g in config.low_cost_groups
    ]
    return variable_options + low_cost_group_options


@app.callback(
    Output("chart_options_tab", "children"),
    Output("chart_options_div", "style"),
    Output("chart_x_variable_options_div", "style"),
    Output("chart_region_div", "style"),
    Output("additional_scenarios_div", "style"),
    Input("chart_options_tab", "value"),
    Input("chart_options", "value"),
)
def tabs_chart(tab_choice, chart_choice):
    """Choose which chart tabs to display."""
    tabs = chart_tab_div_children(chart_choice)
    styles = tab_styles(
        tab_choice, options=["chart", "x_variable", "region", "scenarios"]
    )
    return tabs, *styles


@app.callback(
    Output("chart_x_bin_div", "style"), Input("chart_options", "value")
)
@calls.log
def toggle_bins(chart_type):
    """Show the bin size option under the chart."""

    style = {"display": "none"}
    if chart_type == "binned":
        style = {"margin-left": "10px"}
    return style


@app.callback(
    Output("options", "style"),
    Output("minimizing_scenarios", "style"),
    Output("pca_scenarios", "style"),
    Output("scenario_selection_tabs", "style"),
    Output("toggle_options", "children"),
    Output("toggle_options", "style"),
    Input("toggle_options", "n_clicks"),
    Input("scenario_selection_tabs", "value"),
)
@calls.log
def toggle_options(click, selection_ind):
    """Toggle options on/off."""

    scenario_styles = [{"display": "none"} for _ in range(3)]
    tabs_style = {"display": "none"}
    button_children = "Options: Off"
    button_style = BUTTON_STYLES["off"]

    click = click or 0
    if click % 2 == 1:
        scenario_styles[int(selection_ind)] = {"margin-bottom": "50px"}
        tabs_style = {
            "width": "92%",
            "margin-left": "53px",
            "margin-right": "10px",
        }
        button_children = "Options: On"
        button_style = BUTTON_STYLES["on"]

    return *scenario_styles, tabs_style, button_children, button_style


@app.callback(
    Output("recalc_tab_options", "style"),
    Output("recalc_a_options", "style"),
    Output("recalc_b_options", "style"),
    Input("recalc_tab", "value"),
    Input("recalc_scenario", "value"),
)
def toggle_recalc_tab(recalc, scenario):
    """Toggle the recalc options on and off."""
    tab_style = {}
    recalc_a_style = {}
    recalc_b_style = {}

    # Toggle all options
    if recalc == "off":
        tab_style = {"display": "none"}
    if scenario == "scenario_a":
        recalc_b_style = {"display": "none"}
    else:
        recalc_a_style = {"display": "none"}

    return tab_style, recalc_a_style, recalc_b_style


@app.callback(
    Output("scenario_b_div", "style"),
    Input("difference", "value"),
    Input("mask", "value"),
)
@calls.log
def toggle_scenario_b(difference, mask):
    """Show scenario b if the difference option is on."""

    if difference == "on":
        style = {}
    elif mask == "on":
        style = {}
    else:
        style = {"display": "none"}
    return style


# @app.callback(
#     [
#         Output("scenario_a_specs", "children"),
#         Output("scenario_b_specs", "children"),
#     ],
#     [
#         Input("scenario_a", "value"),
#         Input("scenario_b", "value"),
#         Input("project", "value"),
#     ],
# )
# @calls.log
# def scenario_specs(scenario_a, scenario_b, project):
#     """Output the specs association with a chosen scenario."""
#     # Return a blank space if no parameters entry found
#     params = Config(project).parameters
#     if not params:
#         specs1 = ""
#         specs2 = ""
#     else:
#         if "least_cost" not in scenario_a:
#             scenario_a = os.path.basename(scenario_a).replace("_sc.csv", "")
#             specs1 = build_specs(scenario_a, project)
#         else:
#             specs1 = build_spec_split(scenario_a, project)

#         if "least_cost" not in scenario_b:
#             scenario_b = os.path.basename(scenario_b).replace("_sc.csv", "")
#             specs2 = build_specs(scenario_b, project)
#         else:
#             specs2 = build_spec_split(scenario_b, project)

#     return specs1, specs2
