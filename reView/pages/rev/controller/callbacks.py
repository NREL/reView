# pylint: disable=too-many-lines
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
import tempfile

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from reView.app import app
from reView.components.callbacks import (
    capacity_print,
    display_selected_tab_above_map,
)
from reView.components.logic import tab_styles
from reView.components.map import Map, Title
from reView.layout.options import (
    CHART_OPTIONS,
    COLOR_OPTIONS,
    COLOR_Q_OPTIONS,
)
from reView.layout.styles import TABLET_STYLE
from reView.pages.rev.controller.element_builders import Plots
from reView.pages.rev.controller.selection import (
    choose_scenario,
    get_variable_options,
)
from reView.pages.rev.model import (
    apply_all_selections,
    apply_filters,
    cache_chart_tables,
    cache_map_data,
    cache_table,
    cache_timeseries,
    calc_least_cost,
    ReCalculatedData
)
from reView.utils.bespoke import BespokeUnpacker
from reView.utils.constants import SKIP_VARS
from reView.utils.functions import (
    convert_to_title,
    callback_trigger,
    read_file,
    strip_rev_filename_endings,
    to_geo
)
from reView.utils.config import Config
from reView.utils import calls

logger = logging.getLogger(__name__)


COMMON_CALLBACKS = [
    capacity_print(id_prefix="rev"),
    display_selected_tab_above_map(id_prefix="rev"),
]
DEFAULT_PROJECT = "PR100 - Forecasts"


def build_scenario_dropdowns(groups, dropid=None, multi=False, dynamic=False,
                             typeid="a"):
    """Return list of dropdown options for a project's file selection."""
    # This will be a list of dash dropdowns
    dropdowns = []

    # We'll use alternating colors for readability
    colors = ["#b4c9e0", "#e3effc"]

    # Loop through each group name and options set
    for ind, (group, options) in enumerate(groups.items()):
        # Set style
        color = colors[ind % 2]
        style = {"background-color": color, "margin-right": "-1px"}   
        if ind == 0:
            style["border-top-left-radius"] = "5px"   
        if ind == len(groups) - 1:
            style["border-bottom-left-radius"] = "5px"   

        # Set dropid (is this needed?)
        if dynamic:
            dropid={"index": ind, "type": f"filter-dropdown-{typeid}",
                    "name": group}
        else:
            dropid = f"dropdown_{'_'.join(group.split()).lower()}"

        # Build dropdown object
        dropdown = html.Div(
            children=[
                html.Div(
                    children=html.P(group),
                    className="three columns",
                    style=style
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id=dropid,
                            options=options,
                            value=options[0]["value"],
                            optionHeight=75,
                            multi=multi
                        )
                    ],
                    className="nine columns",
                    style={"margin-left": "-10px"},
                ),
            ],
            style={"border-radius": "5px"},
            className="row",
        )

        # Add to list
        dropdowns.append(dropdown)

    # Wrap all dropdowns in a single div
    drop_div = html.Div(children=dropdowns)

    return drop_div


def build_specs(scenario, project):
    """Calculate the percentage of each scenario present."""
    config = Config(project)
    specs = config.parameters
    dct = specs[scenario]
    table = """| Variable | Value |\n|----------|------------|\n"""
    for variable, value in dct.items():
        row = f"| {variable} | {value} |\n"
        table = table + row
    table = dcc.Markdown(table)
    return table


def build_spec_split(path, project):
    """Calculate the percentage of each scenario present."""
    df = cache_table(project, y_var="capacity", x_var="mean_lcoe", path=path)
    scenarios, counts = np.unique(df["scenario"], return_counts=True)
    total = df.shape[0]
    percentages = [counts[i] / total for i in range(len(counts))]
    percentages = [round(p * 100, 4) for p in percentages]
    pdf = pd.DataFrame({"p": percentages, "s": scenarios})
    pdf = pdf.sort_values("p", ascending=False)
    table = """| Scenario | Percentage |\n|----------|------------|\n"""
    for _, row in pdf.iterrows():
        row = f"| {row['s']} | {row['p']}% |\n"
        table = table + row
    table = dcc.Markdown(table)
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


def composite_fname(paths, composite_function, composite_variable):
    """Attempt to make an intuitive composite tag."""
    # Simplify paths to names
    paths.sort()
    names = [Path(path).name for path in paths]
    names = [strip_rev_filename_endings(name) for name in names]

    # If it's less than 12 paths, use words. If not use hash
    if len(paths) < 12:
        # Remove any repeating elements in the adjusted names
        parts = [list(name.split("_")) for name in names]
        parts = [sublst for lst in parts for sublst in lst]
        repeats = [part for part in parts if parts.count(part) > 1]
        repeats = np.unique(repeats)
        new_names = []
        for repeat in repeats:
            for name in names:
                new_names.append(name.replace(repeat, ""))

        # Join name parts for the tag
        tag = "".join(new_names)
        if tag.startswith("_"):
            tag = tag[1:]
    else:
        tag = hashlib.sha1(str.encode(str(paths))).hexdigest()

    # Build full file name
    identifier = f"{composite_function}_{composite_variable}_{tag}"
    fname = f"composite_{identifier}_supply-curve.csv"

    return fname


def fig_to_df(fig):
    """Return a data frame version of a plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objs._figure.Figure
        A plotly figure with data.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas data frame.
    """
    # Get axis titles
    xtitle = fig.layout["xaxis"]["title"]["text"]
    ytitle = fig.layout["yaxis"]["title"]["text"]

    # Get the data
    subdfs = []
    for item in fig.data:
        group = item.legendgroup.replace(" ", "_").lower()
        subdf = pd.DataFrame({"Group": group, xtitle: item.x, ytitle: item.y})
        subdfs.append(subdf)
    df = pd.concat(subdfs)
    df = df.dropna()

    return df


def filter_files(project, filters, options):
    """Convert div of scenario filter dropdowns to key-value pairs.

    Parameters
    ----------
    project : str
        Name of reV project.
    filters : list
        List of user selected values associated with `options`.

    Returns
    -------
    dict : The key-value pairs of input parameters and user selections.
    """
    # Get the options data frame from the configuration object
    config = Config(project)
    if config.options is None:
        raise ValueError(f"{project} has no variable options CSV.")

    # Simplify the div dictionary 
    filters = dict(zip(options, filters))

    # Apply filters to the options data frame
    options = config.options.copy()
    for col, value in filters.items():
        if value != "all":
            options = options[options[col] == value]

    return list(options["file"].values)


def files_to_dropdown(files, typeid="a"):
    """Convert a list of files to a list of dropdowns."""
    names = [Path(file).name for file in files]
    names = [strip_rev_filename_endings(name) for name in names]
    file_dict = dict(zip(names, files))
    scenario_options = [
        {"label": key, "value": os.path.expanduser(file)}
        for key, file in file_dict.items()
    ]

    return scenario_options


@calls.log
def options_chart_type(project, y_var=None):
    """Add characterization plot option, if necessary."""
    options = CHART_OPTIONS
    if project:
        config = Config(project)
        if config.characterization_cols:
            if y_var in config.characterization_cols:
                options = [CHART_OPTIONS[-1]]
    return options


@app.callback(
    Output("recalc_table_div", "style"),
    Input("project", "value"),
    Input("toggle_options", "n_clicks")
)
@calls.log
def disable_recalc(project, __):
    """Disable recalculate option based on config."""
    # Get config
    config = Config(project)

    # Disable entire table
    div_style = {}
    if not config.parameters:
        div_style = {"display": "none"}

    return div_style


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
    Output("download_chart", "data"),
    Input("download_info_chart", "children"),
    prevent_initial_call=True,
)
@calls.log
def download_chart(chart_info):
    """Download csv file."""
    info = json.loads(chart_info)
    if info["tmp_path"] is None:
        raise PreventUpdate
    src = info["tmp_path"]
    dst = info["path"]
    df = read_file(src)
    os.remove(src)
    return dcc.send_data_frame(df.to_csv, dst, index=False)


# pylint: disable=too-many-locals
@app.callback(
    Output("download_map", "data"),
    Input("rev_map_download_button", "n_clicks"),
    State("map_signal", "children"),
    State("project", "value"),
    State("rev_map", "selectedData"),
    State("rev_chart", "selectedData"),
    State("variable", "value"),
    State("rev_chart_x_var_options", "value"),
    State("rev_chart_options", "value"),
    prevent_initial_call=True
)
@calls.log
def download_map(__, signal, project, map_selection, chart_selection, y_var,
                 x_var, chart_type):
    """Download geopackage file from map."""
    # Retrieve the data frame
    signal_dict = json.loads(signal)
    df = cache_map_data(signal_dict)
    df = apply_all_selections(
        df=df,
        signal_dict=signal_dict,
        project=project,
        chart_selection=chart_selection,
        map_selection=map_selection,
        y_var=y_var,
        x_var=x_var,
        chart_type=chart_type
    )

    # Create the table name
    name = os.path.splitext(os.path.basename(signal_dict["path"]))[0]
    if signal_dict["path2"]:
        name2 = os.path.splitext(os.path.basename(signal_dict["path2"]))[0]
        name = f"{name}_vs_{name2}_diff"

    # Build geopackage and send it
    layer = f"review_{name}_{y_var}"
    with tempfile.NamedTemporaryFile() as tmp:
        dst = tmp.name
        fname = layer + ".gpkg"
        to_geo(df, dst, layer)

        return dcc.send_file(dst, fname)


@app.callback(
    Output("rev_chart_options", "options"),
    Output("rev_chart_options", "value"),
    Input("submit", "n_clicks"),
    State("project", "value"),
    State("variable", "value"),
    State("rev_chart_options", "value")
)
@calls.log
def dropdown_chart_types(_, project, y_var, current_option):
    """Add characterization plot option, if necessary."""
    options = options_chart_type(project, y_var)
    if len(options) == 1:
        value = options[0]["value"]
    else:
        value = current_option
    return options, value


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
def dropdown_colors(__, variable, project, signal, ___):
    """Provide qualitative color options for categorical data."""
    # To figure out if we need to update we need these
    signal_dict = json.loads(signal)
    if not project:
        project = signal_dict["project"]
    old_variable = signal_dict["y"]
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


@app.callback(
    Output("composite_target", "options"),
    Output("composite_target", "value"),
    Input("composite_options", "children"),
    State("project", "value"),
)
def dropdown_composite_targets(scenario_options, project):
    """Set the minimizing target options."""
    logger.debug("Setting minimizing target options")
    config = Config(project)
    path = choose_scenario(scenario_options, config)
    target_options = []
    if path and os.path.exists(path):
        data = read_file(path, nrows=1)
        columns = [c for c in data.columns if c.lower() not in SKIP_VARS]
        titles = {col: convert_to_title(col) for col in columns}
        titles.update(config.titles)
        if titles:
            for key, val in titles.items():
                target_options.append({"label": val, "value": key})

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
def dropdown_projects(__, ___):
    """Update project options."""
    # Open config json
    project_options = [
        {"label": project, "value": project}
        # pylint: disable=not-an-iterable
        for project in Config.sorted_projects
    ]

    if DEFAULT_PROJECT is None:
        default_project = project_options[0]["value"]
    else:
        default_project = DEFAULT_PROJECT

    return project_options, default_project


@app.callback(
    Output("composite_plot_value", "options"),
    Input("composite_options", "children"),
    State("project", "value"),
)
@calls.log
def dropdown_composite_plot_options(scenario_options, project):
    """Set the minimizing plot options."""
    logger.debug("Setting minimizing plot options")
    config = Config(project)
    path = choose_scenario(scenario_options, config)
    plot_options = [
        {"label": "Scenario", "value": "scenario"}
    ]
    if path and os.path.exists(path):
        data = read_file(path, nrows=1)
        columns = [c for c in data.columns if c.lower() not in SKIP_VARS]
        titles = {col: convert_to_title(col) for col in columns}
        titles.update(config.titles)
        if titles:
            for key, val in titles.items():
                plot_options.append({"label": val, "value": key})

    return plot_options


# pylint: disable=too-many-locals
@app.callback(
    Output("scenario_dropdown_a", "options"),
    Output("scenario_dropdown_b", "options"),
    Output("rev_additional_scenarios", "options"),
    Output("rev_additional_scenarios_time", "options"),
    Output("scenario_dropdown_a", "value"),
    Output("scenario_dropdown_b", "value"),
    Output("scenario_dropdown_a", "placeholder"),
    Output("scenario_dropdown_b", "placeholder"),
    Input("project", "value"),
    Input({"type": "filter-dropdown-a", "index": ALL, "name": ALL}, "value"),
    Input({"type": "filter-dropdown-b", "index": ALL, "name": ALL}, "value"),
    Input("url", "pathname"),
    Input("options_tabs", "value"),
    State({"type": "filter-dropdown-a", "index": ALL, "name": ALL}, "id"),
    State("submit", "n_clicks"),
)
@calls.log
def dropdown_scenarios(
        project,
        filters_a,
        filters_b,
        _,
        __,
        filter_ids,
        ___,
    ):
    """Update the scenario options given a project."""
    # Find all available project files
    config = Config(project)

    # Separate the output files, let's put those at the end
    files = [str(file) for file in config.files.values()]
    originals = [file for file in files if "review_outputs" not in file]
    outputs = [file for file in files if "review_outputs" in file]
    originals.sort()
    outputs.sort()

    # If filters are provided, use them to downselect files
    if config.options is not None:
        # Get simple list of option names
        options = [entry["name"] for entry in filter_ids]

        # Get filter list of files
        files_a = filter_files(project, filters_a, options)
        files_b = filter_files(project, filters_b, options)
        files_a += outputs
        files_b += outputs

    # If not return all files
    else:
        # Recombine orginal and output paths
        files_a = files_b = originals + outputs

    # Convert to dropdowns
    group_a = files_to_dropdown(files_a, typeid="a")
    group_b = files_to_dropdown(files_b, typeid="b")

    # Populate with initial values
    if not group_a:
        value_a = None
    else:
        value_a = files_a[0]
    if not group_b:
        value_b = None
    else:
        value_b = files_b[0]

    # Placeholder for when all files are filtered out
    placeholder = "All files filtered out"

    return group_a, group_b, group_a, group_a, value_a, value_b, placeholder, placeholder


# pylint: disable=no-member,too-many-locals
@app.callback(
    Output("composite_scenarios", "options"),
    Input("url", "pathname"),
    Input("project", "value"),
    Input("composite_variable", "value"),
    Input({"type": "filter-dropdown-c", "index": ALL, "name": ALL}, "value"),
    State({"type": "filter-dropdown-c", "index": ALL, "name": ALL}, "id"),
    State("submit", "n_clicks"),
)
@calls.log
def dropdown_scenarios_composite(
        url, 
        project,
        composite_variable,
        filters,
        filter_ids,
        __
    ):
    """Update the options given a project."""
    logger.debug("URL: %s", url)
    config = Config(project)

    # Separate the output files, let's put those at the end
    files = [str(file) for file in config.files.values()]
    files = [file for file in files if "review_outputs" not in file]
    files.sort()

    # If filters are provided, use them to downselect files
    if config.options is not None:
        # Get simple list of option names
        options = [entry["name"] for entry in filter_ids]

        # Get filter list of files
        files = filter_files(project, filters, options)

    group = files_to_dropdown(files, typeid="c")

    return group


@app.callback(
    Output("composite_variable", "options"),
    Input("project", "value")
)
@calls.log
def dropdown_composite_variables(project):
    """Set the minimizing variable options."""
    logger.debug("Setting variable target options")
    config = Config(project)
    scenario_a = config.files[next(iter(config.files))]
    variable_options = get_variable_options(project, scenario_a, None, {})
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
    Output("variable", "options"),
    Output("variable", "value"),
    Output("filter_variables_1", "options"),
    Output("filter_variables_2", "options"),
    Output("filter_variables_3", "options"),
    Output("filter_variables_4", "options"),
    Input("url", "href"),
    Input("scenario_dropdown_a", "value"),
    Input("scenario_dropdown_b", "value"),
    Input("project", "value"),
    State("scenario_b_div", "style"),
    State("variable", "value")
)
@calls.log
def dropdown_variables(
        __,
        scenario_a,
        scenario_b,
        project,
        b_div,
        old_variable
):
    """Update variable dropdown options."""
    # Scrape variable options from entire div
    variable_options = get_variable_options(
        project, scenario_a, scenario_b, b_div
    )
    if not variable_options:
        print("NO VARIABLE OPTIONS FOUND!")
        variable_options = [{"label": "None", "value": "None"}]

    # If the old choice is available, use that
    values = [o["value"] for o in variable_options]
    if old_variable in values:
        value = old_variable
    else:
        value = "capacity"

    return (
        variable_options,
        value,
        variable_options,
        variable_options,
        variable_options,
        variable_options,
    )


@app.callback(
    Output("rev_chart_x_var_options", "options"),
    Output("rev_chart_x_var_options", "value"),
    Input("submit", "n_clicks"),
    Input("rev_chart_options", "value"),
    Input("scenario_dropdown_a", "value"),
    Input("scenario_dropdown_b", "value"),
    State("scenario_b_div", "style"),
    State("project", "value"),
)
@calls.log
def dropdown_x_variables(
    _, chart_type, scenario_a, scenario_b, b_div, project
):
    """Return dropdown options for x variable."""
    logger.debug("Setting X variable options")
    if chart_type == "char_histogram":
        config = Config(project)
        variable_options = [
            {"label": config.titles.get(x, convert_to_title(x)), "value": x}
            for x in config.characterization_cols
        ]
        val = variable_options[0]["value"]
    else:
        variable_options = get_variable_options(
            project, scenario_a, scenario_b, b_div
        )
        val = "capacity"

    if not variable_options:
        variable_options = [{"label": "None", "value": "None"}]
        val = "None"

    return variable_options, val


# @app.callback(
#     Output("rev_additional_scenarios", "options"),
#     Output("rev_additional_scenarios_time", "options"),
#     Input("url", "pathname"),
#     Input("submit", "n_clicks"),
#     Input("project", "value"),
# )
# @calls.log
# def dropdowns_additional_scenarios(url, __, project):
#     """Update the additional scenarios options given a project."""
#     logger.debug("URL: %s", url)

#     # We need the project configuration
#     if not project:
#         config = Config(sorted(Config.projects)[0])
#     else:
#         config = Config(project)

#     # Find the files
#     scenario_outputs_path = config.directory.joinpath("review_outputs")
#     scenario_outputs = [
#         str(f) for f in scenario_outputs_path.glob("least*.csv")
#     ]
#     scenario_originals = [str(file) for file in config.files.values()]
#     files = scenario_originals + scenario_outputs
#     files.sort()
#     names = []
#     for file in files:
#         name = os.path.basename(file)
#         name = strip_rev_filename_endings(name)
#         name = " ".join([n.capitalize() for n in name.split("_")])
#         names.append(name)
#     file_list = dict(zip(names, files))

#     scenario_options = [
#         {"label": key, "value": os.path.expanduser(file)}
#         for key, file in file_list.items()
#     ]

#     if not scenario_options:
#         scenario_options = [{"label": "None", "value": None}]

#     least_cost_options = []
#     for key, file in file_list.items():
#         if file in config.files.values():
#             option = {"label": key, "value": str(file)}
#             least_cost_options.append(option)

#     # Now build the h5 list
#     scenario_options_h5 = [
#         {"label": key, "value": os.path.expanduser(file)}
#         for key, file in file_list.items()
#         if file.endswith(".h5")
#     ]

#     return scenario_options, scenario_options_h5


# pylint: disable=too-many-arguments,unused-argument
@app.callback(
    Output("rev_chart", "figure"),
    Output("rev_chart_loading", "style"),
    Output("download_info_chart", "children"),
    Input("rev_chart_options", "value"),
    Input("rev_map", "selectedData"),
    Input("rev_chart_point_size", "value"),
    Input("chosen_map_options", "children"),
    Input("rev_map_color_min", "value"),
    Input("rev_map_color_max", "value"),
    Input("rev_chart_x_bin", "value"),
    Input("rev_chart_alpha", "value"),
    Input("rev_chart_download_button", "n_clicks"),
    Input("map_signal", "children"),
    State("rev_chart", "selectedData"),
    State("project", "value"),
    State("rev_chart", "relayoutData"),
    State("map_function", "value")
)
@calls.log
def figure_chart(
    chart_type,
    map_selection,
    point_size,
    map_options,
    user_ymin,
    user_ymax,
    bins,
    alpha,
    download,
    signal,
    chart_selection,
    project,
    chart_view,
    map_func
):
    """Make one of a variety of charts."""
    # Unpack the signal
    signal_dict = json.loads(signal)

    # Initial page load project
    if not project:
        project = signal_dict["project"]

    # Collect minimum needed inputs
    x_var = signal_dict["x"]
    if x_var == "None":
        x_var = "capacity"
    y_var = signal_dict["y"]
    project = signal_dict["project"]

    # Get the data frames
    dfs = cache_chart_tables(signal_dict)

    # Return empty alert
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
                df=df,
                signal_dict=signal_dict,
                project=project,
                chart_selection=None,
                map_selection=map_selection,
                y_var=y_var,
                x_var=x_var,
                chart_type=chart_type
            ) for k, df in dfs.items()
        }

    # Build Title
    title_builder = Title(dfs, signal_dict, y_var, project,
                          chart_selection=chart_selection)
    title = title_builder.chart_title

    # This might be a difference
    if signal_dict["path2"] and os.path.isfile(signal_dict["path2"]):
        y_var = dfs[next(iter(dfs))].columns[-1]
    else:
        y_var = signal_dict["y"]

    # Build plotting object
    plotter = Plots(
        project,
        dfs,
        plot_title=title,
        point_size=point_size,
        user_scale=(user_ymin, user_ymax),
        alpha=alpha,
    )
    fig = plotter.figure(chart_type, x_var, y_var, bins)

    # Save download information
    tmp_path = None
    if "rev_chart_download_button" in callback_trigger():
        config = Config(project)
        tmp_dir = config.directory.joinpath("temp")
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = str(tmp_dir.joinpath("temp_chart.csv"))
        df = fig_to_df(fig)
        df.to_csv(tmp_path, index=False)

    # Package returns
    loading_style = {"margin-right": "500px"}
    download_info = json.dumps(
        {"path": "review_chart_data.csv", "tmp_path": tmp_path}
    )

    return fig, loading_style, download_info


# pylint: disable=too-many-arguments,too-many-locals,unused-argument
@app.callback(
    Output("rev_map", "figure"),
    Output("rev_mapcap", "children"),
    Output("rev_map_loading", "style"),
    Input("rev_map_basemap_options", "value"),
    Input("rev_map_color_options", "value"),
    Input("rev_chart", "selectedData"),
    Input("rev_map_point_size", "value"),
    Input("rev_map_rev_color", "n_clicks"),
    Input("rev_map_color_min", "value"),
    Input("rev_map_color_max", "value"),
    Input("rev_map", "selectedData"),
    Input("rev_map", "clickData"),
    Input("map_signal", "children"),
    State("project", "value"),
    State("map_function", "value"),
    State("rev_chart_x_var_options", "value"),
    State("rev_chart_options", "value"),
)
@calls.log
def figure_map(
    basemap,
    color,
    chart_selection,
    point_size,
    reverse_color_clicks,
    color_ymin,
    color_ymax,
    map_selection,
    map_click,
    signal,
    project,
    map_function,
    x_var,
    chart_type
):
    """Make the scatter plot map."""
    # Unpack signal and retrieve data frame
    signal_dict = json.loads(signal)
    df = cache_map_data(signal_dict)

    # Initial page load project
    if not project:
        project = signal_dict["project"]

    # This might be a difference
    if signal_dict["path2"] and os.path.isfile(signal_dict["path2"]):
        y_var = df.columns[-1]
    else:
        y_var = signal_dict["y"]

    # This could also be a modal category
    if y_var in Config(project).characterization_cols:
        y_var += "_mode"

    # Apply user selections
    df = apply_all_selections(
        df=df,
        signal_dict=signal_dict,
        project=project,
        chart_selection=chart_selection,
        map_selection=map_selection,
        y_var=y_var,
        x_var=x_var,
        chart_type=chart_type
    )

    # Apply filters again for characterizations
    filters = signal_dict["filters"]
    df = apply_filters(df, filters)

    if "clickData" in callback_trigger() and "turbine_y_coords" in df:
        unpacker = BespokeUnpacker(df, map_click)
        df = unpacker.unpack_turbines()

    # Use demand counts if available
    if "demand_connect_count" in df:
        color_var = "demand_connect_count"
    else:
        color_var = y_var

    title_builder = Title(df, signal_dict, color_var, project,
                          map_selection=map_selection)
    title = title_builder.map_title

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
        demand_data=None
    )
    figure = map_builder.figure(
        point_size=point_size,
        reverse_color=reverse_color_clicks % 2 == 1,
    )
    mapcap = df[["sc_point_gid", "capacity"]].to_dict()

    # Package returns
    mapcap = json.dumps(mapcap)
    loading_style = {"margin-right": "500px"}

    return figure, mapcap, loading_style


# pylint: disable=too-many-arguments,unused-argument
@app.callback(
    Output("rev_time", "figure"),
    Output("rev_time_loading", "style"),
    Input("map_signal", "children"),
    Input("rev_time_trace_options_tab", "value"),
    Input("rev_time_period_options_tab", "value"),
    Input("rev_variable_time", "value"),
    Input("rev_additional_scenarios_time", "value"),
    Input("rev_chart", "selectedData"),
    Input("rev_map", "selectedData"),
    Input("rev_map", "clickData"),
    State("project", "value")
)
@calls.log
def figure_timeseries(
        signal,
        trace_type,
        time_period,
        variable,
        additional_scenarios,
        chart_selection,
        map_selection,
        map_click,
        project
    ):
    """Render timeseries plots if possible."""
    # read in signal
    signal_dict = json.loads(signal)
    file = signal_dict["path"]
    if not file.endswith(".h5"):
        raise PreventUpdate

    # Initial page load project
    if not project:
        project = signal_dict["project"]

    # Collect all requested datasets
    datasets = {}
    files = [file]
    if additional_scenarios:
        files += additional_scenarios

    # Read in timeseries data
    for file in files:
        name = strip_rev_filename_endings(Path(file).name)
        try:
            data = cache_timeseries(
                file,
                map_selection,
                chart_selection,
                map_click
            )
        except (KeyError, ValueError):
            raise PreventUpdate
        datasets[name] = data

    # Build Title
    title_builder = Title(datasets, signal_dict, variable, project)
    title = title_builder.chart_title

    # Create figure
    plotter = Plots(
        project,
        datasets=datasets,
        plot_title=title,
        point_size=10,
        user_scale=(0, 1),
        alpha=1,
    )
    fig = plotter.figure(
        chart_type="timeseries",
        y_var=variable,
        trace_type=trace_type,
        time_period=time_period
    )

    return fig, None


@app.callback(
    Output("recalc_a_options", "children"),
    Input("project", "value"),
    Input("scenario_dropdown_a", "value"),
    State("recalc_table_store", "children"),
)
@calls.log
def options_recalc_a(project, scenario, recalc_table):
    """Update the drop down options for each scenario."""
    config = Config(project)
    data = ReCalculatedData(config)
    recalc_table = json.loads(recalc_table)
    scenario = os.path.basename(scenario).replace("_sc.csv", "")

    if scenario not in config.scenarios:
        raise PreventUpdate

    if not config.parameters:
        raise PreventUpdate

    scenario = os.path.basename(scenario).replace("_sc.csv", "")
    table = recalc_table["scenario_a"]
    if scenario not in config.parameters:
        raise PreventUpdate

    original_table = data.original_parameters(scenario)

    children = [
        # FCR A
        html.Div(
            [
                html.P(
                    "FCR % (A): ",
                    className="three columns",
                    style={"height": "60%"},
                ),
                dcc.Input(
                    id="fcr1",
                    type="number",
                    className="nine columns",
                    style={"height": "60%"},
                    value=table["fcr"],
                    placeholder=original_table["fcr"],
                ),
            ],
            className="row",
        ),
        # CAPEX A
        html.Div(
            [
                html.P(
                    "CAPEX $/KW (A): ",
                    className="three columns",
                    style={"height": "60%"},
                ),
                dcc.Input(
                    id="capex1",
                    type="number",
                    className="nine columns",
                    style={"height": "60%"},
                    value=table["capex"],
                    placeholder=original_table["capex"],
                ),
            ],
            className="row",
        ),
        # OPEX A
        html.Div(
            [
                html.P(
                    "OPEX $/KW (A): ",
                    className="three columns",
                    style={"height": "60%"},
                ),
                dcc.Input(
                    id="opex1",
                    type="number",
                    className="nine columns",
                    style={"height": "60%"},
                    value=table["opex"],
                    placeholder=original_table["opex"],
                ),
            ],
            className="row",
        ),
        # Losses A
        html.Div(
            [
                html.P(
                    "Losses % (A): ",
                    className="three columns",
                    style={"height": "60%"},
                ),
                dcc.Input(
                    id="losses1",
                    type="number",
                    className="nine columns",
                    value=table["losses"],
                    placeholder=original_table["losses"],
                    style={"height": "60%"},
                ),
            ],
            className="row",
        ),
    ]

    return children


@app.callback(
    Output("recalc_b_options", "children"),
    Input("project", "value"),
    Input("scenario_dropdown_b", "value"),
    State("recalc_table_store", "children"),
)
@calls.log
def options_recalc_b(project, scenario, recalc_table):
    """Update the drop down options for each scenario."""
    config = Config(project)
    data = ReCalculatedData(config)
    recalc_table = json.loads(recalc_table)
    scenario = os.path.basename(scenario).replace("_sc.csv", "")

    if scenario not in config.scenarios:
        raise PreventUpdate

    if not config.parameters:
        raise PreventUpdate

    scenario = os.path.basename(scenario).replace("_sc.csv", "")
    table = recalc_table["scenario_b"]
    original_table = data.original_parameters(scenario)

    children = [
        # FCR B
        html.Div(
            [
                html.P(
                    "FCR % (B): ",
                    className="three columns",
                    style={"height": "60%"},
                ),
                dcc.Input(
                    id="fcr2",
                    type="number",
                    className="nine columns",
                    style={"height": "60%"},
                    value=table["fcr"],
                    placeholder=original_table["fcr"],
                ),
            ],
            className="row",
        ),
        # CAPEX B
        html.Div(
            [
                html.P(
                    "CAPEX $/KW (B): ",
                    className="three columns",
                    style={"height": "60%"},
                ),
                dcc.Input(
                    id="capex2",
                    type="number",
                    className="nine columns",
                    style={"height": "60%"},
                    value=table["capex"],
                    placeholder=original_table["capex"],
                ),
            ],
            className="row",
        ),
        # OPEX B
        html.Div(
            [
                html.P(
                    "OPEX $/KW (B): ",
                    className="three columns",
                    style={"height": "60%"},
                ),
                dcc.Input(
                    id="opex2",
                    type="number",
                    className="nine columns",
                    style={"height": "60%"},
                    value=table["opex"],
                    placeholder=original_table["opex"],
                ),
            ],
            className="row",
        ),
        # Losses B
        html.Div(
            [
                html.P(
                    "Losses % (B): ",
                    className="three columns",
                    style={"height": "60%"},
                ),
                dcc.Input(
                    id="losses2",
                    type="number",
                    className="nine columns",
                    value=table["losses"],
                    placeholder=original_table["losses"],
                    style={"height": "60%"},
                ),
            ],
            className="row",
        ),
    ]

    return children


@app.callback(
    Output("rev_chart_data_signal", "children"),
    Input("variable", "value"),
    Input("rev_chart_x_var_options", "value"),
    Input("rev_map_state_options", "value"),
)
@calls.log
def retrieve_chart_tables(y_var, x_var, state):
    """Store the signal used to get the set of tables needed for the chart."""
    signal = json.dumps([y_var, x_var, state])
    return signal


# pylint: disable=invalid-name
@app.callback(
    Output("filter_store", "children"),
    Input("submit", "n_clicks"),
    Input("project", "value"),
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
def retrieve_filters(__, ___, var1, var2, var3, var4, q1, q2, q3, q4):
    """Retrieve filter variable names and queries."""

    variables = [var1, var2, var3, var4]
    queries = [q1, q2, q3, q4]

    filters = []
    for i, var in enumerate(variables):
        if var and queries[i]:
            filters.append(" ".join([var, queries[i]]))

    return json.dumps(filters)


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches
# pylint: disable=too-many-statements
@app.callback(
    Output("map_signal", "children"),
    Output("pca_plot_1", "clickData"),
    Output("pca_plot_2", "clickData"),
    Input("submit", "n_clicks"),
    Input("rev_map_state_options", "value"),
    Input("rev_map_region_options", "value"),
    Input("rev_chart_options", "value"),
    Input("rev_submit_additional_scenarios", "n_clicks"),
    State("rev_chart_x_var_options", "value"),
    State("rev_additional_scenarios", "value"),
    State("filter_store", "children"),
    State("pca_plot_1", "clickData"),
    State("pca_plot_2", "clickData"),
    State("project", "value"),
    State("variable", "value"),
    State("difference", "value"),
    State("mask", "value"),
    State("recalc_table_store", "children"),
    State("recalc_tab", "value"),
    State("difference_units", "value"),
    State("scenario_dropdown_a", "value"),
    State("scenario_dropdown_b", "value"),
    State("composite_div", "style"),
    State("composite_scenarios", "value"),
    State("composite_variable", "value"),
    State("composite_function", "value"),
    State("composite_plot_value", "value"),
    State("pca_plot_map_value", "value"),
    State("pca_plot_region", "value"),
)
@calls.log
def retrieve_signal(
    __,
    states,
    regions,
    ___,
    ____,
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
    scenario_a,
    scenario_b,
    composite_div_style,
    composite_scenarios,
    composite_variable,
    composite_function,
    composite_plot_value,
    pca_plot_value,
    pca_plot_region,
):
    """Create signal for sharing data between map and chart with dependence."""
    # The first call is sometimes triggered before the initial project is set
    if project is None:
        raise PreventUpdate

    config = Config(project)
    trigger = callback_trigger()

    if scenario_a == "placeholder":
        files = list(config.files.values())
        files.sort()
        scenario_a = files[0]

    # Unpack recalc table
    if recalc_table:
        recalc_table = json.loads(recalc_table)

    lowest_scenario_open = (
        composite_div_style
        and composite_div_style.get("display") != "none"
    )

    if lowest_scenario_open:
        if composite_variable in config.low_cost_groups:
            paths = [
                config.directory.joinpath(file)
                for file in config.low_cost_groups[composite_variable]
            ]
        else:
            paths = [Path(path) for path in composite_scenarios]

        fname = composite_fname(paths, composite_function, composite_variable)

        # Build full paths and create the target file
        dst = config.directory.joinpath("review_outputs", fname)
        dst.parent.mkdir(parents=True, exist_ok=True)
        calc_least_cost(paths, dst, composite_function=composite_function,
                        composite_variable=composite_variable)

        if composite_plot_value == "Variable":
            y = composite_variable
        else:
            y = composite_plot_value

        if not x:
            x = "capacity"

        signal = {
            "filters": [],
            "mask": "off",
            "path": str(dst),
            "path2": None,
            "project": project,
            "recalc": recalc,
            "recalc_table": recalc_table,
            "added_scenarios": [],
            "regions": regions,
            "diff_units": diff_units,
            "states": states,
            "x": x,
            "y": y,
        }

    else:

        # Prevent the first trigger when difference is off
        if "scenario_b" in trigger and diff == "off":
            raise PreventUpdate

        # Prevent the first trigger when mask is off
        if "mask" in trigger and mask == "off":
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
            path = os.path.expanduser(scenario_a)
            if diff == "off" and mask == "off":
                path2 = None
            else:
                path2 = os.path.expanduser(scenario_b)

        logger.debug("path = %s", path)
        logger.debug("path2 = %s", path2)

        if scenarios:
            scenarios = [os.path.expanduser(path) for path in scenarios]

        # Pack up the filters
        if filter_store:
            filters = json.loads(filter_store)
        else:
            filters = []

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
            "diff_units": diff_units,
            "states": states,
            "x": x,
            "y": y,
        }

    return json.dumps(signal), None, None


@app.callback(
    Output("recalc_table_store", "children"),
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
@calls.log
def retrieve_recalc_parameters(
    fcr1, capex1, opex1, losses1, fcr2, capex2, opex2, losses2, __
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
    Output("scenario_a_specs", "children"),
    Output("scenario_b_specs", "children"),
    Output("scenario_a_specs", "style"),
    Output("scenario_b_specs", "style"),
    Input("scenario_dropdown_a", "value"),
    Input("scenario_dropdown_b", "value"),
    State("project", "value"),
)
@calls.log
def scenario_specs(scenario_a, scenario_b, project):
    """Output the specs association with a chosen scenario."""
    # Project might be None on initial load
    if not project:
        raise PreventUpdate

    # Return a blank space if no parameters entry found
    config = Config(project)
    params = config.parameters

    # Infer the names
    path_lookup = {str(value): key for key, value in config.files.items()}
    name_a = path_lookup[scenario_a]
    name_b = path_lookup[scenario_b]

    specs_a = ""
    specs_b = ""
    style_a = {}
    style_b = {}
    if name_a in params:
        style_a = {"overflow-y": "auto", "height": "300px", "width": "94%"}
        specs_a = build_specs(name_a, project)
    if "least_cost" in scenario_a:
        style_a = {"overflow-y": "auto", "height": "300px", "width": "94%"}
        specs_a = build_spec_split(scenario_a, project)
    if name_b in params:
        style_b = {"overflow-y": "auto", "height": "300px", "width": "94%"}
        specs_b = build_specs(name_b, project)
    if "least_cost" in scenario_b:
        style_b = {"overflow-y": "auto", "height": "300px", "width": "94%"}
        specs_b = build_spec_split(scenario_b, project)

    return specs_a, specs_b, style_a, style_b


@app.callback(
    Output("rev_chart_options_tab", "children"),
    Output("rev_chart_options_div", "style"),
    Output("rev_chart_x_variable_options_div", "style"),
    Output("rev_additional_scenarios_div", "style"),
    Input("rev_chart_options_tab", "value"),
    Input("rev_chart_options", "value"),
)
def tabs_chart(tab_choice, chart_choice):
    """Choose which chart tabs to display."""
    tabs = chart_tab_div_children(chart_choice)
    styles = tab_styles(
        tab_choice, options=["chart", "x_variable", "scenarios"]
    )
    return tabs, *styles


@app.callback(
    Output("rev_chart_x_bin_div", "style"), Input("rev_chart_options", "value")
)
@calls.log
def toggle_bins(chart_type):
    """Show the bin size option under the chart."""

    if chart_type in {"binned", "histogram", "char_histogram"}:
        return {}
    return {"display": "none"}


@app.callback(
    Output("toggle_options", "children"),
    Output("options_label", "style"),
    Output("options_div", "is_open"),
    Input("toggle_options", "n_clicks"),
    State("options_div", "is_open"),
)
@calls.log
def toggle_options(click, is_open):
    """Toggle options on/off."""
    click = click or 0

    is_open = not is_open
    if is_open:
        options_label = {"display": "none"}
        button_children = "Hide"
    else:
        options_label = {
            "float": "left",
            "margin-left": "20px",
            "margin-bottom": "-25px",
        }
        button_children = "Show"

    return button_children, options_label, is_open


@app.callback(
    Output("options", "style"),
    Output("composite_div", "style"),
    Input("options_tabs", "value"),
)
@calls.log
def toggle_options_tabs(choice):
    """Toggle toptions tab style."""
    options_style = composite_style = {"display": "none"}
    if choice == "0":
        options_style = {"margin-bottom": "1px"}
    else:
        composite_style = {"margin-bottom": "1px"}

    return options_style, composite_style


@app.callback(
    Output("recalc_tab_options", "style"),
    Output("recalc_a_options", "style"),
    Output("recalc_b_options", "style"),
    Input("recalc_tab", "value"),
    Input("recalc_scenario", "value"),
)
def toggle_recalc_tab(recalc, scenario):
    """Toggle the recalc options on and off."""
    # Set target dictionaries
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
    Output("rev_chart_below_options", "is_open"),
    Input("rev_chart_below_options_button", "n_clicks"),
    State("rev_chart_below_options", "is_open"),
)
@calls.log
def toggle_rev_chart_below_options(n_clicks, is_open):
    """Open or close chart below options."""
    if n_clicks:
        return not is_open
    return is_open


@app.callback(
    Output("rev_map_below_options", "is_open"),
    Input("rev_map_below_options_button", "n_clicks"),
    State("rev_map_below_options", "is_open"),
)
@calls.log
def toggle_rev_map_below_options(n_clicks, is_open):
    """Open or close map below options."""
    if n_clicks:
        return not is_open
    return is_open


@app.callback(
    Output("rev_additional_scenarios_time", "style"),
    Input("rev_time_trace_options_tab", "value")
)
@calls.log
def toggle_timeseries_above_options(trace):
    """Open or close map below options."""
    if trace == "bar":
        style_a = {"display": "none"}
    else:
        style_a = {}
    return style_a


@app.callback(
    Output("rev_time_below_options", "is_open"),
    Input("rev_time_below_options_button", "n_clicks"),
    State("rev_time_below_options", "is_open"),
)
@calls.log
def toggle_timeseries_below_options(n_clicks, is_open):
    """Open or close map below options."""
    if n_clicks:
        return not is_open
    return is_open


@app.callback(
    Output("scenario_a_filter_div", "style"),
    Output("scenario_b_filter_div", "style"),
    Output("composite_filter_div", "style"),
    Output("scenario_a_filters", "children"),
    Output("scenario_b_filters", "children"),
    Output("composite_filters", "children"),
    Input("project", "value"),
)
@calls.log
def toggle_scenario_filters(project):
    """Toggle scenario selection filters if project has variable CSV."""
    def build_options(odf, typeid="a"):
        """Build a list of available options."""
        cols = [col for col in odf.columns if col not in ["name", "file"]]

        # Build the filtering dictionary
        groups = {}
        df = odf.copy()
        for i, col in enumerate(cols):
            # pylint: disable=unsubscriptable-object
            options = df[col].unique()
            options = ["all"] + list(options)

            dropdown_options = []
            for option in options:
                label = str(option)
                dropdown_options.append({"label": label, "value": option})

            groups[col] = dropdown_options

        group = build_scenario_dropdowns(groups, dynamic=True, typeid=typeid)

        return group

    # By default, hide the filtering dropdowns
    style_a = style_b = style_c = {"display": "none"}

    # Get the config object and check if it has filtering options
    config = Config(project)
    group_a = group_b = group_c = []
    if config.options is not None:

        # If so, set the filtering div styles
        style_a = style_b = style_c = {}

        # Get just the target columns
        odf = config.options

        # Convert these dictionaries into dropdowns
        group_a = build_options(odf, typeid="a")
        group_b = build_options(odf, typeid="b")
        group_c = build_options(odf, typeid="c")

    return style_a, style_b, style_c, group_a, group_b, group_c


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


@app.callback(
    Output("timeseries", "style"),
    Input("submit", "n_clicks"),
    Input("url", "pathname"),
    Input("scenario_dropdown_a", "value"),
)
@calls.log
def toggle_timeseries(_, ___, scenario):
    """Toggle the timeseries component on/off in response to chose dataset."""
    if scenario.endswith(".h5"):
        style = {"margin-top": "50px"}
    else:
        style = {"display": "none"}
    return style
