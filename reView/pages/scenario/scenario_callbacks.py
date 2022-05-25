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

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from sklearn.metrics import DistanceMetric

from reView.app import app
from reView.layout.styles import BUTTON_STYLES, TABLET_STYLE, RC_STYLES
from reView.layout.options import (
    CHART_OPTIONS,
    COLOR_OPTIONS,
    COLOR_Q_OPTIONS,
)
from reView.pages.scenario.element_builders import (
    build_title,
    Map
)
from reView.pages.scenario.scenario_data import (
    apply_all_selections,
    Plots,
    calc_least_cost,
    cache_table,
    cache_chart_tables,
)
from reView.utils.constants import SKIP_VARS
from reView.utils.functions import convert_to_title
from reView.utils.config import Config
from reView.utils import args

logger = logging.getLogger(__name__)


DIST_METRIC = DistanceMetric.get_metric("haversine")
# PCA_DF = pd.read_csv(
#     Path.home() / "review_datasets" / "hydrogen_pca" / "pca_df_300_sites.csv"
# )


def all_files_from_selection(selected_options, config):
    """_summary_

    Parameters
    ----------
    selected_options : _type_
        _description_
    config : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    try:
        df = config.options.copy()
    except AttributeError:
        raise ValueError("Missing project options csv!") from None

    for k, v in selected_options.items():
        df = df[df[k] == v["value"]]
    return df



# def build_pca_plot(
#     color, x, y, z, camera=None, ymin=None, ymax=None, state="CONUS"
# ):
#     """Build a Plotly pca plot."""

#     # Create hover text
#     # if units == "category":
#     #     df["text"] = (
#     #         df["county"]
#     #         + " County, "
#     #         + df["state"]
#     #         + ": <br>   "
#     #         + df[y].astype(str)
#     #         + " "
#     #         + units
#     #     )
#     # else:
#     #     extra_str = ""
#     #     if "hydrogen_annual_kg" in df:
#     #         extra_str += (
#     #             "<br>    H2 Supply:    "
#     #             + df["hydrogen_annual_kg"].apply(lambda x: f"{x:,}")
#     #             + " kg    "
#     #         )
#     #     if "dist_to_selected_load" in df:
#     #         extra_str += (
#     #             "<br>    Dist to load:    "
#     #             + df["dist_to_selected_load"].apply(lambda x: f"{x:,.2f}")
#     #             + " km    "
#     #         )

#     #     df["text"] = (
#     #         df["county"]
#     #         + " County, "
#     #         + df["state"]
#     #         + ":"
#     #         + extra_str
#     #         + f"<br>    {convert_to_title(y)}:   "
#     #         + df[y].round(2).astype(str)
#     #         + " "
#     #         + units
#     #     )

#     # marker = dict(
#     #     color=df[y],
#     #     colorscale=pcolor,
#     #     cmax=None if ymax is None else float(ymax),
#     #     cmin=None if ymin is None else float(ymin),
#     #     opacity=1.0,
#     #     reversescale=rev_color,
#     #     size=point_size,
#     #     colorbar=dict(
#     #         title=dict(
#     #             text=units,
#     #             font=dict(
#     #                 size=15, color="white", family="New Times Roman"
#     #             ),
#     #         ),
#     #         tickfont=dict(color="white", family="New Times Roman"),
#     #     ),
#     # )

#     # Create data object
#     # figure = px.scatter_mapbox(
#     #     data_frame=df,
#     #     lon="longitude",
#     #     lat="latitude",
#     #     custom_data=["sc_point_gid", "print_capacity"],
#     #     hover_name="text",
#     # )

#     principal_df = PCA_DF[PCA_DF.State == state]
#     features = [
#         "electrolyzer_size_ratio",
#         "wind_cost_multiplier",
#         "fcr",
#         "water_cost_multiplier",
#         "pipeline_cost_multiplier",
#         "electrolyzer_size_mw",
#         "electrolyzer_capex_per_mw",
#     ]
#     range_color = (
#         None if ymin is None else float(ymin),
#         None if ymax is None else float(ymax),
#     )
#     figure = px.scatter_3d(
#         principal_df,
#         x=x,
#         y=y,
#         z=z,
#         color=color,
#         range_color=range_color,
#         size_max=15,
#         # marker=dict(size=3, symbol="circle"),
#         hover_name=principal_df[color],
#         hover_data=features,
#         custom_data=["file"],
#         # text=[f for f in principal_df['file']]
#     )
#     # figure.update_traces(marker=marker)
#     if camera is not None:
#         figure.update_layout(scene_camera=camera)
#     # figure = make_subplots(rows=1, cols=2,
#     #                        shared_xaxes=True,
#     #                        shared_yaxes=True,
#     #                        specs=[[
#     #                            {'type': 'surface'},
#     #                            {'type': 'surface'}
#     #                         ]],
#     #                     # vertical_spacing=0.02
#     #                     )
#     # scatter = go.Scatter3d(x = principal_df['pc1'],
#     #                        y = principal_df['pc2'],
#     #                        z = principal_df['pc3'],
#     #                        mode ='markers',
#     #                        marker = dict(
#     #                         size = 12,
#     #                         color = principal_df[y],
#     #                         # colorscale ='Viridis',
#     #                         # opacity = 0.8
#     #                     )
#     #                     )
#     # scatter2 = go.Scatter3d(x = principal_df['pc1'],
#     #                        y = principal_df['pc2'],
#     #                        z = principal_df['pc3'],
#     #                        mode ='markers',
#     #                        marker = dict(
#     #                         size = 12,
#     #                         color = principal_df[y],
#     #                         # colorscale ='Viridis',
#     #                         # opacity = 0.8
#     #                     )
#     #                     )

#     # figure.add_trace(scatter, row=1, col=1)
#     # figure.add_trace(scatter2, row=1, col=2)

#     # figure.update_layout(height=600, width=600,
#     #                      title_text="Stacked Subplots with Shared X-Axes")

#     # Update the layout
#     # layout_ = build_map_layout(
#     #     title, basemap, showlegend, ymin, ymax
#     # )
#     # figure.update_layout(**layout_)

#     return figure


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




def chart_tab_styles(tab_choice):
    """Set correct tab styles for the chosen option."""
    styles = [{"display": "none"}] * 4
    order = ["chart", "x_variable", "region", "scenarios"]
    idx = order.index(tab_choice)
    styles[idx] = {"width": "100%", "text-align": "center"}
    return styles


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

    # Add x variable otpion if needed
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


def choose_scenario(scenario_options, config):
    """_summary_

    Parameters
    ----------
    scenario_options : _type_
        _description_
    config : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    selected_options = parse_selection(scenario_options)
    logger.debug("selected_options = %s", selected_options)
    if not selected_options:
        # TODO what if `files` is empty?
        # How to tell user no files found in dir?
        return list(config.files.values())[0]

    if "Scenario" in selected_options:
        return selected_options["Scenario"]["value"]

    return file_for_selections(selected_options, config)


def closest_demand_to_coords(selection_coords, demand_data):
    """_summary_

    Parameters
    ----------
    selection_coords : _type_
        _description_
    demand_data : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    demand_coords = demand_data[["latitude", "longitude"]].values
    demand_coords_rad = np.radians(demand_coords)
    out = DIST_METRIC.pairwise(np.r_[selection_coords, demand_coords_rad])
    load_center_ind = np.argmin(out[0][1:])
    return load_center_ind


def closest_load_center(load_center_ind, demand_data):
    """_summary_

    Parameters
    ----------
    load_center_ind : _type_
        _description_
    demand_data : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    demand_coords = demand_data[["latitude", "longitude"]].values
    demand_coords_rad = np.radians(demand_coords)
    load_center_info = demand_data.iloc[load_center_ind]
    load_center_coords = demand_coords_rad[load_center_ind]
    load = load_center_info[["load"]].values[0]
    return load_center_coords, load


def file_for_selections(selected_options, config):
    """_summary_

    Parameters
    ----------
    selected_options : _type_
        _description_
    config : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    row = all_files_from_selection(selected_options, config)
    logger.debug("row = %s", row)
    if "file" in row:
        logger.debug("file = %s", row["file"].values[0])
        return row["file"].values[0]

    name = row["name"].values[0]
    logger.debug("name = %s", name)
    file = config.files.get(f"{name}")
    logger.debug("file = %s", file)
    return file


def filter_points_by_demand(df, load_center_coords, load):
    """_summary_

    Parameters
    ----------
    df : _type_
        _description_
    load_center_coords : _type_
        _description_
    load : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    sc_coords = df[["latitude", "longitude"]].values
    sc_coords = np.radians(sc_coords)
    load_center_coords = np.array(load_center_coords).reshape(-1, 2)
    out = DIST_METRIC.pairwise(load_center_coords, sc_coords)
    # print(out.shape, df.shape)
    df["dist_to_selected_load"] = out.reshape(-1) * 6373.0
    df["selected_load_pipe_lcoh_component"] = (
        df["pipe_lcoh_component"]
        / df["dist_to_h2_load_km"]
        * df["dist_to_selected_load"]
    )
    df["selected_lcoh"] = (
        df["no_pipe_lcoh_fcr"] + df["selected_load_pipe_lcoh_component"]
    )
    df = df.sort_values("selected_lcoh")
    df["h2_supply"] = df["hydrogen_annual_kg"].cumsum()
    where_inds = np.where(df["h2_supply"] >= load)[0]
    # print(f'{load=}')
    max_supply = df["h2_supply"].max()
    # print(f'{max_supply=}')
    # print(f'{where_inds=}')
    if where_inds.size > 0:
        final_ind = np.where(df["h2_supply"] >= load)[0].min() + 1
        df = df.iloc[0:final_ind]
    # print(f'{df=}')
    return df


def options_chart_type(project):
    """Add characterization plot option, if necessary."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    if Config(project).characterizations_cols:
        return CHART_OPTIONS
    else:
        return CHART_OPTIONS[:-1]


def parse_selection(scenario_options):
    """_summary_

    Parameters
    ----------
    scenario_options : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if scenario_options is None:
        return {}
    selected_options = {}
    scenarios_div = scenario_options["props"]["children"]
    for option_div in scenarios_div:
        option_name_div = option_div["props"]["children"][0]
        title_div = option_name_div["props"]["children"][0]
        selection_name = title_div["props"]["children"]

        selection_div = option_div["props"]["children"][1]
        dropdown_div = selection_div["props"]["children"][0]
        selection_value = dropdown_div["props"]  # ["value"]
        selected_options[selection_name] = selection_value

    return selected_options


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
                            optionHeight=75
                        )
                    ],
                    className=class_names[-1],
                ),
            ],
            className="row",
            style={"background-color": color},
        )

        dropdowns.append(dropdown)

    dropdiv = html.Div(
        children=dropdowns,
        style={"border": "4px solid #1663b5", "padding": "2px"},
    )

    return dropdiv


def scrape_variable_options(project, scenario_a_options, scenario_b_options,
                            b_div):
    """Retrieve appropriate variable list."""
    config = Config(project)
    path = choose_scenario(scenario_a_options, config)
    variable_options = []
    if path and os.path.exists(path):
        columns = pd.read_csv(path, nrows=1).columns
        if b_div.get("display") != "none":
            path2 = choose_scenario(scenario_b_options, config)
            if path2 and os.path.exists(path2):
                columns2 = pd.read_csv(path2, nrows=1).columns
                columns = [c for c in columns if c in columns2]
        columns = [c for c in columns if c.lower() not in SKIP_VARS]
        titles = {col: convert_to_title(col) for col in columns}
        config_titles = {k: v for k, v in config.titles.items() if k in titles}
        titles.update(config_titles)
        if titles:
            for k, v in titles.items():
                variable_options.append({"label": v, "value": k})

    return variable_options


@app.callback(
    Output("capacity_print", "children"),
    Output("site_print", "children"),
    Input("mapcap", "children"),
    Input("map", "selectedData")
)
def capacity_print(mapcap, mapsel):
    """Calculate total remaining capacity after all filters are applied."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    # Calling this from make_map where the chartsel has already been applied
    nsites = ""
    capacity = ""
    if mapcap:
        df = pd.DataFrame(json.loads(mapcap))
        if not df.empty:
            if mapsel:
                gids = [p.get("customdata", [None])[0] for p in mapsel["points"]]
                df = df[df["sc_point_gid"].isin(gids)]
            nsites = "{:,}".format(df.shape[0])
            total_capacity = df["print_capacity"].sum()
            if total_capacity >= 1_000_000:
                capacity = f"{round(total_capacity / 1_000_000, 4)} TW"
            else:
                capacity = f"{round(total_capacity / 1_000, 4)} GW"

    return capacity, nsites


@app.callback(
    Output("recalculate_with_new_costs", "hidden"),
    Input("project", "value"),
    Input("toggle_options", "n_clicks")
)
def disable_recalculate_with_new_costs(project, __):
    return not Config(project).parameters


@app.callback(
    Output("map_function_div", "hidden"),
    Output("map_function", "value"),
    Input("project", "value"),
    Input("toggle_options", "n_clicks")
)
def disable_mapping_function_dev(project, __):
    return Config(project).demand_data is None, "None"


@app.callback(
    Output("chart_options", "options"),
    Input("project", "value"),
)
def dropdown_chart_types(project):
    """Add characterization plot option, if necessary."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    return options_chart_type(project)


@app.callback(
    Output("color_options", "options"),
    Output("color_options", "value"),
    Input("submit", "n_clicks"),
    State("variable", "value"),
    State("project", "value"),
    State("map_signal", "children"),
    State("color_options", "value")
)
def dropdown_colors(submit, variable, project, signal, old_value):
    """Provide qualitative color options for categorical data."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    # To figure out if we need to update we need these
    if not signal:
        raise PreventUpdate  # @IgnoreException
    old_variable = json.loads(signal)["y"]
    config = Config(project)
    units = config.units.get(variable, "")
    old_units = config.units.get(old_variable, "")

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


@app.callback(
    Output("minimizing_scenario_options", "children"),
    Input("url", "pathname"),
    Input("project", "value"),
    Input("minimizing_variable", "value"),
    State("submit", "n_clicks")
)
def dropdown_minimizing_scenarios(url, project, minimizing_variable, n_clicks):
    """Update the options given a project."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    # Catch the trigger
    logger.debug("URL: %s", url)
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    # We need the project configuration
    config = Config(project)

    if config.options is not None:
        groups = {}

        for col in config.options.columns:
            if col in {"name", "file"} or col == minimizing_variable:
                continue
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
    State("project", "value")
)
def dropdown_minimizing_targets(scenario_options, project):
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
    State("submit", "n_clicks")
)
def dropdown_projects(pathname, n_clicks):
    """Update project options."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    # Open config json
    project_options = [
        {"label": project, "value": project} for project in Config.projects
    ]
    return project_options, project_options[0]["value"]


@app.callback(
    Output("minimizing_plot_value", "options"),
    Input("minimizing_scenario_options", "children"),
    State("project", "value"),
)
def dropdown_minimizing_plot_options(scenario_options, project):
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
    State("submit", "n_clicks")
)
def dropdown_scenarios(url, project, n_clicks):
    """Update the options given a project."""
    # Store argument values
    logger.debug("URL: %s", url)
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    # We need the project configuration
    config = Config(project)

    if config.options is not None:
        groups = {}

        for col in config.options.columns:
            if col in {"name", "file"}:
                continue
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
def dropdown_variables(url, scenario_a_options, scenario_b_options, b_div,
                       project):
    """Update variable dropdown options."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

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
    Output("chart_xvariable_options", "options"),
    Output("chart_xvariable_options", "value"),
    Input("scenario_a_options", "children"),
    Input("scenario_b_options", "children"),
    Input("scenario_b_div", "style"),
    Input("chart_options", "value"),
    State("project", "value")
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
    State("submit", "n_clicks")
)
def dropdowns_additional_scenarios(
    url,
    project,
    n_clicks,
):
    """Update the additional scenarios options given a project."""
    # Store argument values
    logger.debug("URL: %s", url)
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

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
    Input("map", "selectedData"),
    Input("chart_point_size", "value"),
    Input("chosen_map_options", "children"),
    Input("chart_region", "value"),
    Input("map_color_min", "value"),
    Input("map_color_max", "value"),
    Input("chart_xbin", "value"),
    Input("chart_alpha", "value"),
    State("chart", "selectedData"),
    State("project", "value"),
    State("chart", "relayoutData"),
    State("map_function", "value")
)
def figure_chart(signal, chart, mapsel, point_size, op_values, region, uymin,
                 uymax, bin_size, alpha, chartsel, project, chartview,
                 map_func):
    """Make one of a variety of charts."""
    trig = dash.callback_context.triggered[0]["prop_id"]
    args.setargs(**locals())

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
    if mapsel:
        dfs = {
            k: apply_all_selections(
                df, map_func, project, chartsel, mapsel, clicksel=None
            )[0]
            for k, df in dfs.items()
        }

    plotter = Plots(
        project,
        dfs,
        plot_title=build_title(dfs, signal_dict, chartsel=chartsel),
        point_size=point_size,
        user_scale=(uymin, uymax),
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
    Output("map", "figure"),
    Output("mapcap", "children"),
    Output("map", "clickData"),
    Output("map_loading", "style"),
    Input("map_signal", "children"),
    Input("basemap_options", "value"),
    Input("color_options", "value"),
    Input("chart", "selectedData"),
    Input("map_point_size", "value"),
    Input("rev_color", "n_clicks"),
    Input("map_color_min", "value"),
    Input("map_color_max", "value"),
    Input("map", "selectedData"),
    Input("map", "clickData"),
    State("project", "value"),
    State("map_function", "value")
)
def figure_map(signal, basemap, color, chartsel, point_size, rev_color, uymin,
               uymax, mapsel, clicksel, project, mapfunc):
    """Make the scatterplot map."""
    # Store arguments for later
    trigger = dash.callback_context.triggered[0]["prop_id"]
    args.setargs(**locals())

    # Build figure
    map_builder = Map(
        basemap=basemap,
        chartsel=chartsel,
        clicksel=clicksel,
        color=color,
        mapsel=mapsel,
        mapfunc=mapfunc,
        point_size=point_size,
        project=project,
        rev_color=rev_color,
        signal=signal,
        trigger=trigger,
        uymin=uymin,
        uymax=uymax
    )
    figure = map_builder.figure
    mapcap = map_builder.mapcap

    return figure, json.dumps(mapcap), None, {"float": "left"}


# @app.callback(
#     [
#         Output("pca_plot_1", "figure"),
#         Output("pca_plot_2", "figure"),
#         # Output("mapcap", "children"),
#         # Output("pca_plot_1", "clickData"),
#     ],
#     [
#         Input("pca_scenarios", "style"),
#         Input("pca_plot_value_1", "value"),
#         Input("pca_plot_value_2", "value"),
#         Input("pca_plot_1", "relayoutData"),
#         Input("pca_plot_2", "relayoutData"),
#         # Input("map_signal", "children"),
#         # Input("basemap_options", "value"),
#         # Input("color_options", "value"),
#         # Input("chart", "selectedData"),
#         # Input("map_point_size", "value"),
#         # Input("rev_color", "n_clicks"),
#         Input("pca1_color_min", "value"),
#         Input("pca1_color_max", "value"),
#         Input("pca2_color_min", "value"),
#         Input("pca2_color_max", "value"),
#         Input("pca_plot_axis1", "value"),
#         Input("pca_plot_axis2", "value"),
#         Input("pca_plot_axis3", "value"),
#         Input("pca_plot_region", "value"),
#         # Input("pca_plot_1", "clickData"),
#     ],
#     # [
#     # State("project", "value"),
#     # State("map", "relayoutData"),
#     # State("map_function", "value"),
#     # ],
# )
# def make_pca_plot(
#     pca_scenarios_style,
#     pca_plot_value_1,
#     pca_plot_value_2,
#     data_plot_one,
#     data_plot_two,
#     uymin1,
#     uymax1,
#     uymin2,
#     uymax2,
#     pca_plot_axis1,
#     pca_plot_axis2,
#     pca_plot_axis3,
#     pca_plot_region
#     # clicksel
# ):
#     """Make the pca plot."""

#     trig = dash.callback_context.triggered[0]["prop_id"]

#     # Don't update if selected data triggered this?
#     logger.debug("PCA TRIGGER: %s", trig)
#     if pca_scenarios_style.get("display") == "none":
#         raise PreventUpdate  # @IgnoreException

#     if pca_plot_value_1 == "None" or pca_plot_value_2 == "None":
#         raise PreventUpdate  # @IgnoreException

#     # print(clicksel)
#     # if clicksel and clicksel.get('points'):
#     #     raise PreventUpdate

#     # df, demand_data = apply_all_selections(
#     #     df, map_project, chartsel, mapsel, clicksel
#     # )

#     logger.debug("Building pca plot")
#     if trig == "pca_plot_1.relayoutData" and data_plot_one:
#         camera = data_plot_one.get("scene.camera")
#     elif trig == "pca_plot_2.relayoutData" and data_plot_two:
#         camera = data_plot_two.get("scene.camera")
#     else:
#         camera = None

#     figure = build_pca_plot(
#         pca_plot_value_1,
#         pca_plot_axis1,
#         pca_plot_axis2,
#         pca_plot_axis3,
#         camera,
#         uymin1,
#         uymax1,
#         pca_plot_region,
#     )
#     figure2 = build_pca_plot(
#         pca_plot_value_2,
#         pca_plot_axis1,
#         pca_plot_axis2,
#         pca_plot_axis3,
#         camera,
#         uymin2,
#         uymax2,
#         pca_plot_region,
#     )

#     return figure, figure2  # , None


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
# def options_recalc_a(project, scenario, recalc_table):
#     """Update the drop down options for each scenario."""
    # # Store argument values
    # trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    # args.setargs(**locals())

#     data = Data(project)
#     recalc_table = json.loads(recalc_table)
#     scenario = os.path.basename(scenario).replace("_sc.csv", "")
#     if scenario not in data.scenarios:
#         raise PreventUpdate

#     if not data.parameters:
#         raise PreventUpdate

#     table = recalc_table["scenario_a"]
#     otable = data.original_parameters(scenario)
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
#                     placeholder=otable["fcr"],
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
#                     placeholder=otable["capex"],
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
#                     placeholder=otable["opex"],
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
#                     placeholder=otable["losses"],
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
# def options_recalc_b(project, scenario, recalc_table):
#     """Update the drop down options for each scenario."""
    # Store argument values
    # trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    # args.setargs(**locals())

#     data = Data(project)
#     recalc_table = json.loads(recalc_table)
#     if scenario not in data.scenarios:
#         raise PreventUpdate

#     if not data.parameters:
#         raise PreventUpdate

#     scenario = os.path.basename(scenario).replace("_sc.csv", "")
#     table = recalc_table["scenario_b"]
#     otable = data.original_parameters(scenario)
#     scenario = os.path.basename(scenario).replace("_sc.csv", "")
#     table = recalc_table["scenario_b"]
#     otable = data.original_parameters(scenario)
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
#                     placeholder=otable["fcr"],
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
#                     placeholder=otable["capex"],
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
#                     placeholder=otable["opex"],
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
#                     placeholder=otable["losses"],
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
    Input("chart_xvariable_options", "value"),
    Input("state_options", "value")
)
def retrieve_chart_tables(y, x, state):
    """Store the signal used to get the set of tables needed for the chart."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

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
    State("filter_4", "value")
)
def retrieve_filters(submit, var1, var2, var3, var4, q1, q2, q3, q4):
    """Retrieve filter variable names and queries."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

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
    Input("state_options", "value"),
    Input("region_options", "value"),
    Input("chart_options", "value"),
    Input("chart_xvariable_options", "value"),
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
    State("pca_plot_region", "value")
)
def retrieve_signal(
    submit,
    states,
    regions,
    chart,
    x,
    scenarios,
    filter_store,
    pca1_clicksel,
    pca2_clicksel,
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
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())
    logger.debug("Trigger: %s", trig)

    # Get/build the value scale table
    config = Config(project)

    # Unpack recalc table
    if recalc_table:
        recalc_table = json.loads(recalc_table)

    lowest_scen_open = (
        minimizing_scenarios_style
        and minimizing_scenarios_style.get("display") != "none"
    )

    if lowest_scen_open:
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
        lchh_path = config.directory / ".review" / fname
        lchh_path.parent.mkdir(parents=True, exist_ok=True)
        # calculator = LeastCost(project)
        calc_least_cost(paths, lchh_path, by=minimizing_target)
        if minimizing_plot_value == "Variable":
            y = "scenario"
        else:
            y = minimizing_plot_value
        signal = {
            "filters": [],
            "mask": "off",
            "path": str(lchh_path),
            "path2": None,
            "project": project,
            "recalc": recalc,
            "recalc_table": recalc_table,
            "added_scenarios": [],
            "regions": regions,
            "states": states,
            "x": x,
            "y": y,
            # "selection_a": {},
            # "selection_b": {},
        }
        return json.dumps(signal), None, None

    # Prevent the first trigger when difference is off
    if "scenario_b" in trig and diff == "off":
        raise PreventUpdate

    # Prevent the first trigger when mask is off
    if "mask" in trig and mask == "off":
        raise PreventUpdate

    if pca1_clicksel and pca1_clicksel.get("points"):
        path = pca1_clicksel["points"][0]["customdata"][0]
        path2 = None
        y = pca_plot_value
        states = [] if pca_plot_region == "CONUS" else [pca_plot_region]
    elif pca2_clicksel and pca2_clicksel.get("points"):
        path = pca2_clicksel["points"][0]["customdata"][0]
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

    # Packup the filters
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
        # "selection_a": parse_selection(scenario_a_options),
        # "selection_b": parse_selection(scenario_b_options),
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
def retrieve_recalc_parameters(fcr1, capex1, opex1, losses1, fcr2, capex2,
                               opex2, losses2, project):
    """Retrive all given recalc values and store them."""
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if "project" == trig:
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


# @app.callback(
#     Output("pca_plot_value_1", "options"),
#     Output("pca_plot_value_1", "value"),
#     Output("pca_plot_value_2", "options"),
#     Output("pca_plot_value_2", "value"),
#     Output("pca_plot_axis1", "options"),
#     Output("pca_plot_axis1", "value"),
#     Output("pca_plot_axis2", "options"),
#     Output("pca_plot_axis2", "value"),
#     Output("pca_plot_axis3", "options"),
#     Output("pca_plot_axis3", "value"),
#     Output("pca_plot_region", "options"),
#     Output("pca_plot_region", "value"),
#     # Input("project", "value"),
#     Input("pca_scenarios", "style"),
# )
# def set_pca_variable_options(
#     # project,
#     pca_scenarios_style,
# ):
#     logger.debug("Setting variable target options")
#     # config = Config(project)
#     variable_options = [{"label": "None", "value": "None"}]
#     axis_options = [
#         {"label": "pc1", "value": "pc1"},
#         {"label": "pc2", "value": "pc2"},
#         {"label": "pc3", "value": "pc3"},
#     ]
#     region_options = [{"label": "CONUS", "value": "CONUS"}]
#     is_showing = (
#         pca_scenarios_style and pca_scenarios_style.get("display") != "none"
#     )
#     if is_showing:
#         variable_options += [
#             {"label": convert_to_title(col), "value": col}
#             for col in PCA_DF.columns
#             if col not in {"pc1", "pc2", "pc3", "file", "State"}
#         ]
#         axis_options += [
#             {"label": convert_to_title(col), "value": col}
#             for col in PCA_DF.columns
#             if col not in {"pc1", "pc2", "pc3", "file", "State"}
#         ]
#         region_options += [
#             {"label": convert_to_title(state), "value": state}
#             for state in PCA_DF.State.unique()
#             if state != "CONUS"
#         ]
#     return (
#         variable_options,
#         variable_options[-1]["value"],
#         variable_options,
#         variable_options[-1]["value"],
#         axis_options,
#         "pc1",
#         axis_options,
#         "pc2",
#         axis_options,
#         "pc3",
#         region_options,
#         "CONUS",
#     )


# @app.callback(
#     Output("pca_plot_map_value", "options"),
#     Output("pca_plot_map_value", "value"),
#     Input("project", "value"),
# )
# def set_pca_plot_options(project):
#     """"""
#     logger.debug("Setting pca plot options")
#     config = Config(project)
#     # TODO: Remove hardcoded path
#     # path = choose_scenario(scenario_options, config)
#     path = "C:\\Users\\ppinchuk\\review_datasets\\hydrogen\\review_pca\\wind_flat_esr01_wcm0_ecpm0_f0035_wcm10_pcm05_nrwal_00.csv"
#     plot_options = [{"label": "Variable", "value": "Variable"}]
#     if path and os.path.exists(path):
#         data = pd.read_csv(path)
#         columns = [c for c in data.columns if c.lower() not in SKIP_VARS]
#         titles = {col: convert_to_title(col) for col in columns}
#         titles.update(config.titles)
#         if titles:
#             for k, v in titles.items():
#                 plot_options.append({"label": v, "value": k})

#     return plot_options, plot_options[-1]["value"]


@app.callback(
    Output("chart_options_tab", "children"),
    Output("chart_options_div", "style"),
    Output("chart_xvariable_options_div", "style"),
    Output("chart_region_div", "style"),
    Output("additional_scenarios_div", "style"),
    Input("chart_options_tab", "value"),
    Input("chart_options", "value")
)
def tabs_chart(tab_choice, chart_choice):
    """Choose which chart tabs to display."""
    tabs = chart_tab_div_children(chart_choice)
    styles = chart_tab_styles(tab_choice)
    return tabs, *styles


@app.callback(
    Output("state_options", "style"),
    Output("region_options", "style"),
    Output("basemap_options_div", "style"),
    Output("color_options_div", "style"),
    Input("map_options_tab", "value")
)
def tabs_map(tab_choice):
    """Choose which map tabs to display."""
    # Styles
    styles = [{"display": "none"}] * 4
    order = ["state", "region", "basemap", "color"]
    idx = order.index(tab_choice)
    styles[idx] = {"width": "100%", "text-align": "center"}
    return styles[0], styles[1], styles[2], styles[3]


@app.callback(
    Output("chart_xbin_div", "style"),
    Input("chart_options", "value")
)
def toggle_bins(chart_type):
    """Show the bin size option under the chart."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    style = {"display": "none"}
    if chart_type == "binned":
        style = {"margin-left": "10px"}
    return style


@app.callback(
        Output("options", "style"),
        Output("minimizing_scenarios", "style"),
        Output("pca_scenarios", "style"),
        Output("scen_selection_tabs", "style"),
        Output("toggle_options", "children"),
        Output("toggle_options", "style"),
        Input("toggle_options", "n_clicks"),
        Input("scen_selection_tabs", "value")
)
def toggle_options(click, selection_ind):
    """Toggle options on/off."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    scen_styles = [{"display": "none"} for _ in range(3)]
    tabs_style = {"display": "none"}
    button_children = "Options: Off"
    button_style = BUTTON_STYLES["off"]

    click = click or 0
    if click % 2 == 1:
        scen_styles[int(selection_ind)] = {"margin-bottom": "50px"}
        tabs_style = {
            "width": "92%",
            "margin-left": "53px",
            "margin-right": "10px",
        }
        button_children = "Options: On"
        button_style = BUTTON_STYLES["on"]

    return *scen_styles, tabs_style, button_children, button_style


@app.callback(
    Output("recalc_tab_options", "style"),
    Output("recalc_a_options", "style"),
    Output("recalc_b_options", "style"),
    Input("recalc_tab", "value"),
    Input("recalc_scenario", "value")
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
        Output("rev_color", "children"),
        Output("rev_color", "style"),
        Input("rev_color", "n_clicks")
)
def toggle_rev_color_button(click):
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


@app.callback(
    Output("scenario_b_div", "style"),
    Input("difference", "value"),
    Input("mask", "value")
)
def toggle_scenario_b(difference, mask):
    """Show scenario b if the difference option is on."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

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
# def scenario_specs(scenario_a, scenario_b, project):
#     """Output the specs association with a chosen scenario."""
    # # Store argument values
    # trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    # args.setargs(**locals())

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


