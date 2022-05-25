# -*- coding: utf-8 -*-
"""Configuration Application.

Created on Sun Aug 23 16:27:25 2020

@author: travis
"""
import ast
import json
import os
import pathos.multiprocessing as mp
from pathlib import Path

from glob import glob

import dash
import numpy as np
import pandas as pd
import tkinter as tk

from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from reView.utils.constants import (
    ORIGINAL_FIELDS,
    TITLES,
    UNITS,
)
from reView import REVIEW_CONFIG_DIR
from reView.utils.config import Config, Process, PROJECT_NAMES
from reView.utils import args


from reView.app import app
from tkinter import filedialog
from tqdm import tqdm

PROJECTS = [{"label": p, "value": p} for p in Config.projects]


# Refactoring this guy....class below not implemented
class Config_Input(Config):
    """Methods for building a project config from user input."""

    def __init__(self, project):
        """Initialize Config_Input object."""
        super().__init__(project)

    def __repr__(self):
        """Return representation string."""
        attrs = ["=".join([k, str(v)]) for k, v in self.__dict__.items()]
        msg = ", ".join(attrs)
        return f"<Config_Input: {msg}>"

    def browser(self, initialdir):
        """Open a file system browser.

        Parameters
        ----------
        initialdir : str
            Path to working directory.
        """

    def groups(self, group_input):
        """Create group dictionary and add to config."""

    def parameters(self, file_inputs):
        """Create parameter dictionary and add to config."""

    def file_df(self, file_inputs):
        """Create file dataframe and add to config."""

    def units(self, unit_inputs):
        """Create unit dictionary and add to config."""


def get_scales(file_df, field_units):
    """Create a value scale dictionary for each field-unit pair."""

    def get_range(args):
        file, fields = args
        ranges = {}
        df = pd.read_csv(file, low_memory=False)
        for field in fields:
            ranges[field] = {}
            if field in df.columns:
                try:
                    values = df[field].dropna()
                    values = values[values != -np.inf]
                    ranges[field]["min"] = values.min()
                    ranges[field]["max"] = values.max()
                except KeyError:
                    print("KeyError")
                    del ranges[field]
            else:
                ranges[field]["min"] = 0
                ranges[field]["max"] = 9999
        return ranges

    # Get all the files
    files = file_df["file"].values
    numbers = [k for k, v in field_units.items() if v != "category"]
    categories = [k for k, v in field_units.items() if v == "category"]

    # Setup numeric scale runs
    arg_list = [[file, numbers] for file in files]
    ranges = []
    with mp.Pool(mp.cpu_count()) as pool:
        for rng in pool.imap(get_range, arg_list):
            ranges.append(rng)

    # Adjust
    def minit(x):
        try:
            m = min([e["min"] for e in x])
            return m
        except:
            return np.nan

    def maxit(x):
        try:
            m = min([e["max"] for e in x])
            return m
        except:
            return np.nan

    rdf = pd.DataFrame(ranges).T
    mins = rdf.apply(minit, axis=1)
    maxes = rdf.apply(maxit, axis=1)

    scales = {}
    for field in rdf.index:
        scales[field] = {}
        vmin = mins[field]
        vmax = maxes[field]
        if isinstance(vmin, np.int64):
            vmin = int(vmin)
            vmax = int(vmax)
        scales[field]["min"] = vmin
        scales[field]["max"] = vmax

    # The LCOES need to be set manually
    for field in scales.keys():
        if "lcoe" in field:
            if scales[field]["min"] < 10:
                scales[field]["min"] = 10
            if scales[field]["max"] > 125:
                scales[field]["max"] = 125

    # Add in qualifier for categorical fields
    for field in categories:
        scales[field] = {"min": "na", "max": "na"}

    return scales


layout = html.Div(
    [
        # Path Name
        dcc.Location(id="/configo_page", refresh=False),
        # Start of the page
        html.H3("Configure Project"),
        # Project name
        html.H5("Project Name"),
        html.Div(
            [
                html.Div(
                    [
                        html.H6("Create New Project:"),
                        dcc.Input(
                            id="project_name",
                            debounce=True,
                            placeholder="Enter project name",
                            style={"width": "100%", "margin-left": "26px"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.H6("Edit Existing Project:"),
                        dcc.Dropdown(
                            id="project_list",
                            options=PROJECTS,
                            placeholder="Select project name",
                            style={"width": "100%", "margin-left": "15px"},
                        ),
                    ]
                ),
            ],
            style={
                "margin-left": "50px",
                "margin-bottom": "15px",
                "width": "35%",
            },
            className="row",
        ),
        # Project directory
        html.H5("Find Project Directory", style={"margin-top": "50px"}),
        html.Div(
            [
                html.Button(
                    children="navigate",
                    title="Navigate the file system for the project directory.",
                    id="proj_nav",
                    n_clicks=0,
                ),
                dcc.Input(
                    id="proj_input",
                    placeholder="/",
                    name="name_test",
                    debounce=True,
                    style={"width": "36%"},
                ),
            ],
            className="row",
            style={"margin-left": "50px"},
        ),
        html.Div(id="project_directory_print"),
        # Dataset paths
        html.H5("Add Data Set(s)", style={"margin-top": "50px"}),
        html.Div(
            [
                html.Button(
                    id="file_nav",
                    children="navigate",
                    title=(
                        "Navigate the file system for a reV supply curve data "
                        "set."
                    ),
                    n_clicks=0,
                ),
                dcc.Input(
                    id="file_pattern",
                    debounce=True,
                    placeholder="Glob Pattern",
                    style={"width": "36%"},
                ),
                html.Div(
                    id="file_groups",
                    style={"margin-bottom": "50px", "width": "50%"},
                ),
            ],
            className="row",
            style={"margin-left": "50px", "margin-bottom": "15px"},
        ),
        # reV configs
        html.H5(
            "reV Configs",
            style={"margin-top": "15px"},
            title=(
                "Find the path to the final pipeline config for each "
                "dataset. This will enable summary parameter print outs "
                "and on-the-fly recalculations for certain variables. "
                "Optional"
            ),
        ),
        html.Div([]),
        # Create Group
        html.H5("Create Groups", style={"margin-top": "50px"}),
        html.Div(
            [
                html.Button(children="Add Group", id="submit_group"),
                dcc.Input(id="group_input", placeholder="Input group name."),
                dcc.Input(
                    id="group_value_input",
                    placeholder="Input Field Name.",
                    style={"width": "27.5%"},
                ),
            ],
            className="row",
            style={"margin-left": "50px", "margin-bottom": "15px"},
        ),
        html.Div(
            id="top_groups", style={"margin-bottom": "50px", "width": "50%"}
        ),
        # Add extra field entries
        html.H5("New Fields Detected"),
        html.Div(
            id="extra_fields",
            style={
                "margin-left": "50px",
                "margin-bottom": "50px",
                "width": "50%",
            },
        ),
        # Submit and trigger configuration build/update
        html.Button(
            id="submit",
            children="submit",
            title="Submit above values and build the project configuration file.",
            n_clicks=0,
        ),
        # Storage
        html.Div(id="proj_dir", style={"display": "none"}, children="/"),
        html.Div(id="groups", children="{}", style={"display": "none"}),
        html.Div(id="files", children="{}", style={"display": "none"}),
        html.Div(id="config", style={"display": "none"}),
    ],
    className="twelve columns",
    style={"margin-bottom": "50px"},
)


def dash_to_pandas(dt):
    """Convert a dash data table to a pandas data frame."""
    df = pd.DataFrame(dt["props"]["data"])
    try:
        df = df.apply(lambda x: ast.literal_eval(x), axis=0)
        msg = "File attributes read successfully."
    except ValueError:
        msg = "File attributes contain missing values."
    return df, msg


def find_unique(files, field):
    """Find all unique values of a field in all files."""

    def single(file, field):
        """Return unique values of a field in one file."""
        df = pd.read_csv(file, usecols=[field])
        return list(df[field].unique())

    uvalues = []
    for file in files:
        uv = single(file, field)
        uvalues = uvalues + uv
    uvalues = list(np.unique(uvalues))
    uvalues = [float(v) if not isinstance(v, str) else v for v in uvalues]
    return uvalues


def navigate(which, initialdir="/"):
    """Browse directory for file or folder paths."""
    # Store argument values
    args.setargs(**locals())

    filetypes = [("ALL", "*"), ("CSV", "*.csv")]

    root = tk.Tk()
    root.withdraw()
    root.geometry("1000x200")
    root.resizable(0, 0)
    back = tk.Frame(master=root, bg="black")
    back.pack_propagate(0)

    root.option_add("*foreground", "black")
    root.option_add("*activeForeground", "black")
    # root.option_add('*font', 'arial')

    # style = ttk.Style(root)
    # style.configure('TLabel', foreground='black', font=font)
    # style.configure('TEntry', foreground='black', font=font)
    # style.configure('TMenubutton', foreground='black', font=font)
    # style.configure('TButton', foreground='black', font=font)

    if which == "files":
        paths = filedialog.askopenfilenames(
            master=root, filetypes=filetypes, initialdir=initialdir
        )
    else:
        paths = filedialog.askdirectory(master=root)

    root.destroy()

    return paths


@app.callback(
    Output("project_list", "options"), [Input("project_name", "value")]
)
def project_list(name):
    """Return or update & return available project list."""
    projects = PROJECT_NAMES
    if name and name not in projects:
        projects = projects + [name]
    projects = [{"label": p, "value": p} for p in projects]
    return projects


@app.callback(
    Output("project_name", "value"), [Input("project_list", "value")]
)
def project_name(selection):
    """Add an existing project selection to the project name entry."""
    return selection


@app.callback(
    Output("proj_input", "placeholder"),
    Output("proj_dir", "children"),
    Output("project_directory_print", "children"),
    Input("project_name", "value"),
    Input("proj_nav", "n_clicks"),
    Input("proj_input", "value")
)
def find_project_directory(name, n_clicks, path):
    """Find the root project directory containing data files."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    if name in PROJECT_NAMES:
        config = Config(name)
        if not path:
            path = str(config.directory)

    if "proj_nav" in trig:
        if n_clicks > 0:
            path = navigate("folders")

    if path:
        if not os.path.exists(path):
            print("Chosen path does not exist.")
    else:
        path = "/"

    sdiv = html.Div(
        id="project_directory",
        children=[
            html.P(path),
        ],
        className="row",
        style={"margin-left": "100px", "margin-bottom": "15px"},
    )

    return path, path, sdiv


@app.callback(
    Output("top_groups", "children"),
    Output("groups", "children"),
    Input("submit_group", "n_clicks"),
    Input("project_name", "value"),
    State("group_input", "value"),
    State("group_value_input", "value"),
    State("groups", "children"),
    State("files", "children")
)
def create_groups(submit, name, group_input, group_values, group_dict, files):
    """Set a group with which to categorize datasets."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    group_dict = json.loads(group_dict)

    field = group_values  # <-------------------------------------------------- Rushing

    if name in group_dict:
        groups = group_dict[name]
    else:
        groups = {}

    if name in PROJECT_NAMES:
        config = Config(name)
        groups = {**config.groups, **groups}

    if group_input:
        if files != "{}" and files != '{"null": []}':
            files = json.loads(files)[name]
            group_values = find_unique(files, field)
            groups[group_input] = {
                "field": field,
                "options": json.dumps(group_values),
            }

    group_dict[name] = groups
    if "" in group_dict:
        del group_dict[""]
    if "null" in group_dict:
        del group_dict["null"]

    df = pd.DataFrame(groups, index=[0]).T
    df = df.reset_index()
    df.columns = ["group", "values"]

    dt = []
    for group, values in groups.items():
        if group:
            reminder = "**{}**: {}".format(group, values)
            sdiv = dcc.Markdown(
                id="{}_options".format(group),
                children=reminder,
                className="row",
                style={"margin-left": "100px", "margin-bottom": "15px"},
            )
            dt.append(sdiv)

    group_dict = json.dumps(group_dict)

    print(dt)

    return dt, group_dict


@app.callback(
    Output("files", "children"),
    Input("file_nav", "n_clicks"),
    Input("file_pattern", "value"),
    Input("project_name", "value"),
    State("proj_dir", "children"),
    State("files", "children"),
)
def add_datasets(n_clicks, pattern, name, initialdir, file_dict):
    """Browse the file system for a list of file paths."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    file_dict = json.loads(file_dict)

    if file_dict == {} or file_dict is None or name not in file_dict:
        files = []
    elif name in file_dict:
        files = [os.path.expanduser(f) for f in file_dict[name]]

    if name in PROJECT_NAMES:
        config = Config(name)
        files = [str(file) for file in config.files.values()]

    if "file_nav" in trig:
        if n_clicks > 0:
            paths = navigate("files", initialdir=initialdir)
            for path in paths:
                if not os.path.exists(path):
                    raise OSError(path + "does not exist.")
                files.append(os.path.join(initialdir, path))
            files = list(np.unique(files))
    elif "file_pattern" in trig:
        fpattern = os.path.expanduser(os.path.join(initialdir, pattern))
        files = glob(fpattern, recursive=True)

    elif "project_name" not in trig:
        raise PreventUpdate

    file_dict[name] = files
    file_dict = json.dumps(file_dict)

    return file_dict


@app.callback(
    Output("file_groups", "children"),
    Input("files", "children"),
    Input("proj_dir", "children"),
    State("project_name", "value"),
)
def set_dataset_table(file_dict, proj_dir, name):
    """For each file, set a group and value from the user inputs above."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    file_dict = json.loads(file_dict)

    if not name:
        return None

    if name not in file_dict:
        raise PreventUpdate

    if not file_dict[name]:
        return None

    if "null" in file_dict:
        del file_dict["null"]

    if file_dict and name in file_dict:
        files = file_dict[name]
        files.sort()
        dropdowns = {}

        # Data Table
        if name in PROJECT_NAMES:
            config = Config(name)
            df = config.data
            if "name" in df:
                del df["name"]
            new_rows = []
            for file in files:
                file = file.replace(os.path.expanduser("~"), "~")
                if file not in df["file"].values:
                    row = df.iloc[0].copy()
                    for key in row.keys():
                        row[key] = None
                    row["file"] = file
                    new_rows.append(row)
            if new_rows:
                ndf = pd.DataFrame(new_rows)
                df = pd.concat([df, ndf])
            df = df.reset_index(drop=True)
        else:
            df = pd.DataFrame({"file": files})

        cols = []
        for col in df.columns:
            entry = {"name": col, "id": col, "editable": True}
            cols.append(entry)

        records = df.to_dict("records")
        values = [record["file"] for record in records]
        values.sort()
        data = [{"file": value} for value in values]

        dt = dash_table.DataTable(
            id="group_table",
            data=data,
            columns=cols,
            editable=True,
            column_selectable="multi",
            row_deletable=True,
            row_selectable="multi",
            page_size=10,
            dropdown=dropdowns,
            style_cell={"textAlign": "left"},
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "rgb(232,240,254)",
                }
            ],
            style_header={
                "backgroundColor": "rgb(22,99,181)",
                "color": "rgb(255,255,255)",
                "fontWeight": "bold",
            },
        )

        return dt


@app.callback(
    Output("extra_fields", "children"),
    Input("files", "children"),
    State("extra_fields", "children"),
    State("project_name", "value"),
)
def find_extra_fields(file_dict, fields, name):
    """Use one of the files to infer extra fields and assign units."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    file_dict = json.loads(file_dict)

    if not name:
        raise PreventUpdate

    if name not in file_dict:
        raise PreventUpdate

    if not file_dict[name]:
        return None

    if "null" in file_dict:
        del file_dict["null"]

    if not file_dict:
        raise PreventUpdate

    files = file_dict[name]

    new_fields = []
    if files:
        for file in files:
            if os.path.isfile(os.path.expanduser(file)):
                columns = pd.read_csv(file, nrows=0).columns
                new_columns = list(set(columns) - set(ORIGINAL_FIELDS))
                new_fields = new_fields + new_columns
        new_fields = list(np.unique(new_fields))
    else:
        raise PreventUpdate

    df = pd.DataFrame(new_fields)
    df["title"] = "N/A"
    df["units"] = "N/A"
    df.columns = ["FIELD", "TITLE", "UNITS"]

    if name in PROJECT_NAMES:
        config = Config(name)
        for field, title in config.titles.items():
            df["TITLE"][df["FIELD"] == field] = title
        for field, unit in config.units.items():
            df["UNITS"][df["FIELD"] == field] = unit

    cols = [{"name": i, "id": i, "editable": True} for i in df.columns]
    cell_styles = [
        {"if": {"column_id": "UNITS"}, "width": "45px"},
        {"if": {"column_id": "FIELD"}, "width": "75px"},
        {"if": {"column_id": "TITLE"}, "width": "75px"},
    ]

    dt = dash_table.DataTable(
        id="field_table",
        data=df.to_dict("records"),
        columns=cols,
        editable=True,
        column_selectable="multi",
        row_selectable="multi",
        row_deletable=True,
        page_size=10,
        style_cell={"textAlign": "left"},
        style_cell_conditional=cell_styles,
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(232,240,254)"}
        ],
        style_header={
            "backgroundColor": "rgb(22,99,181)",
            "color": "rgb(255,255,255)",
            "fontWeight": "bold",
        },
    )

    return dt


@app.callback(
    Output("config", "children"),
    Input("submit", "n_clicks"),
    State("file_groups", "children"),
    State("project_name", "value"),
    State("proj_dir", "children"),
    State("extra_fields", "children"),
    State("groups", "children")
)
def build_config(n_clicks, group_dt, name, directory, fields, groups):
    """Consolidate inputs into a project config and update overall config."""
    # Store argument values
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    args.setargs(**locals())

    if n_clicks == 0:
        raise PreventUpdate

    # Get the file/group and fields data frames
    field_df, fmsg = dash_to_pandas(fields)
    file_df, gmsg = dash_to_pandas(group_dt)
    groups = json.loads(groups)

    # We might need to update the home directory
    if (
        os.path.dirname(file_df["file"].iloc[0]) != directory
    ):  # <------------- This breaks if files are in different folders
        old = os.path.dirname(file_df["file"].iloc[0])
        file_df["file"] = file_df["file"].apply(
            lambda x: x.replace(old, directory)
        )

    # Add just the file name
    fnames = file_df["file"].apply(lambda x: os.path.basename(x))
    file_df["name"] = fnames.apply(lambda x: x.replace(".csv", ""))

    # Combine all titles and units
    units = dict(zip(field_df["FIELD"], field_df["UNITS"]))
    titles = dict(zip(field_df["FIELD"], field_df["TITLE"]))
    field_units = {**units, **UNITS}
    titles = {**titles, **TITLES}
    for field, title in titles.items():
        if title == "N/A":
            titles[field] = field

    # Create Processing object and add additional fields
    print("Applying standard post-processing routines...")
    files = list(file_df["file"].values)
    processor = Process(files=files)
    processor.process()

    # Update field units and titles for sqkm and pct area calcs on counts
    df = pd.read_csv(files[0])
    sqkms = [c for c in df.columns if "sq_km" in c and c != "area_sq_km"]
    for field in sqkms:
        key = "_".join(field.split("_")[:-2])
        title = titles[key].split("-")[0] + " - Area"
        unit = "sqkm"
        titles[field] = title
        field_units[field] = unit
        units[field] = unit

    pcts = [c for c in df.columns if "_pct" in c]
    for field in pcts:
        key = "_".join(field.split("_")[:-1])
        title = titles[key].split("-")[0] + " - Percent Area"
        unit = "%"
        titles[field] = title
        field_units[field] = unit
        units[field] = unit

    # Find value ranges for color scalesondf
    print("Setting min/max values...")
    scales = get_scales(file_df, field_units)

    # For each data frame, if it is missing columns add nans in
    print("Syncing available columns...")
    needed_columns = list(field_units.keys()) + ["nrel_region"]
    for path in tqdm(file_df["file"]):
        dcols = pd.read_csv(path, index_col=0, nrows=0).columns
        if any([c not in dcols for c in needed_columns]):
            df = pd.read_csv(path, low_memory=False)
            for field in needed_columns:
                if field not in df.columns:
                    df[field] = np.nan
            df.to_csv(path, index=False)

    # Convert to a config entry
    config = {
        "data": file_df.to_dict(),
        "directory": directory,
        "groups": groups,
        "parameters": {},  # <------------------------------------------------- add a config input option and infer everything from the pipeline config (SAM, Gen, maybe there are things in agg we'd need).
        "scales": scales,
        "titles": titles,
        "units": units,
    }

    if name in PROJECT_NAMES:
        old_config = Config(name)
        for scenario, params in old_config.parameters.items():
            config["parameters"][scenario] = params

    CONFIG_PATH = Path(REVIEW_CONFIG_DIR) / "new_config.json"
    # Get existing/build new configuration file
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as file:
            full_config = json.load(file)
    else:
        full_config = {}

    # Add in the new entry and save
    full_config[name] = config
    with open(CONFIG_PATH, "w") as file:
        file.write(json.dumps(full_config, indent=4))
