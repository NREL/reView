# -*- coding: utf-8 -*-
"""Config class used for all reView projects.

Created on Sat Aug 15 15:47:40 2020

@author: travis
"""
import logging

from functools import cached_property, lru_cache
from pathlib import Path

import pandas as pd

from reView.utils.constants import UNITS, SCALE_OVERRIDES
from reView.utils.functions import (
    strip_rev_filename_endings,
    load_project_configs,
)

pd.set_option("mode.chained_assignment", None)

logger = logging.getLogger(__name__)

PROJECT_CONFIGS = load_project_configs()
PROJECT_NAMES = list(PROJECT_CONFIGS.keys())
REGIONS = {
    "Pacific": ["Oregon", "Washington"],
    "Mountain": ["Colorado", "Idaho", "Montana", "Wyoming"],
    "Great Plains": [
        "Iowa",
        "Kansas",
        "Missouri",
        "Minnesota",
        "Nebraska",
        "North Dakota",
        "South Dakota",
    ],
    "Great Lakes": ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin"],
    "Northeast": [
        "Connecticut",
        "New Jersey",
        "New York",
        "Maine",
        "New Hampshire",
        "Massachusetts",
        "Pennsylvania",
        "Rhode Island",
        "Vermont",
    ],
    "California": ["California"],
    "Southwest": ["Arizona", "Nevada", "New Mexico", "Utah"],
    "South Central": ["Arkansas", "Louisiana", "Oklahoma", "Texas"],
    "Southeast": [
        "Alabama",
        "Delaware",
        "District of Columbia",
        "Florida",
        "Georgia",
        "Kentucky",
        "Maryland",
        "Mississippi",
        "North Carolina",
        "South Carolina",
        "Tennessee",
        "Virginia",
        "West Virginia",
    ],
}
RESOURCE_CLASSES = {
    "windspeed": {
        "onshore": {
            1: [9.01, 100],
            2: [8.77, 9.01],
            3: [8.57, 8.77],
            4: [8.35, 8.57],
            5: [8.07, 8.35],
            6: [7.62, 8.07],
            7: [7.10, 7.62],
            8: [6.53, 7.10],
            9: [5.90, 6.53],
            10: [0, 5.90],
        },
        "offshore": {
            "fixed": {
                1: [9.98, 100],
                2: [9.31, 9.98],
                3: [9.13, 9.31],
                4: [8.85, 9.13],
                5: [7.94, 8.85],
                6: [7.07, 7.94],
                7: [0, 7.07],
            },
            "floating": {
                1: [10.30, 1000],
                2: [10.01, 10.30],
                3: [9.60, 10.01],
                4: [8.84, 9.60],
                5: [7.43, 8.84],
                6: [5.98, 7.43],
                7: [0, 5.98],
            },
        },
    }
}


def capex(df):
    """Recalculate capital costs if needed input columns are present."""
    capacity = df["capacity"]
    capacity_kw = capacity * 1000

    fcr = df["mean_fixed_charge_rate"]

    unit_om = df["mean_fixed_operating_cost"] / df["mean_system_capacity"]
    om = unit_om * capacity_kw

    mean_cf = df["mean_cf"]
    lcoe = df["mean_lcoe"]
    if "raw_lcoe" in df:
        raw_lcoe = df["raw_lcoe"]
    else:
        raw_lcoe = lcoe.copy()

    cc = (
        (lcoe * (capacity * mean_cf * 8760)) - om
    ) / fcr  # Watch out for economies of scale here
    unit_cc = cc / capacity_kw  # $/kw

    raw_cc = (
        (raw_lcoe * (capacity * mean_cf * 8760)) - om
    ) / fcr  # Watch out for economies of scale here
    raw_unit_cc = raw_cc / capacity_kw  # $/kw

    df["capex"] = cc
    df["unit_capex"] = unit_cc
    df["raw_capex"] = raw_cc
    df["raw_unit_capex"] = raw_unit_cc

    return df



class Config:
    """Class for handling configuration variables."""

    _all_configs = {}
    REQUIREMENTS = {"directory"}

    def __new__(cls, project):
        return cls._all_configs.setdefault(project, super().__new__(cls))

    def __init__(self, project):
        """Initialize plotting object for a reV project."""
        self.project = project
        self._config = None
        self._check_valid_project_name()
        self._set_config()
        self._check_required_keys_exist()

    def __str__(self):
        msg = (
            f"<reView Config object: "
            f"project={self.project!r}, {len(self.files)} files>"
        )
        return msg

    def __repr__(self):
        return f"Config({self.project!r})"

    @property
    def characterizations_cols(self):
        """list: List of column names with characterization info."""
        return self._config.get("characterizations_cols", [])

    @property
    def directory(self):
        """:obj:`pathlib.Path`: Project directory path."""
        return self._extract_fp_from_config("directory")

    @property
    def demand_data(self):
        """Return demand data if it exists."""
        return self._safe_read("demand_file")

    @cached_property
    def files(self):
        """Return a dictionary of scenario with full paths to files."""
        # TODO: If we need better performance, we may consider caching the
        #       output of this in a project-based class-level dictionary so
        #       that it can be quickly accessed when the config class is
        #       reinitialized
        return dict(self._project_files)

    @property
    def groups(self):
        """dict: Groups dictionary."""
        return self._config.get("groups", {})

    @property
    def low_cost_groups(self):
        """dict: Low-cost group options dictionary."""
        return self._config.get("low_cost_groups", {})

    @cached_property
    def options(self):
        """:obj:`pandas.DataFrame` or `None`: Dataframe containing
        variables as column names and values as rows, or `None` if the
        "var_file" key was not specified in the config.
        """
        return self._safe_read(
            "var_file", default_fp=self.directory / "variable_options.csv"
        )

    @property
    def parameters(self):
        """dict: Parameters config dictionary."""
        return self._config.get("parameters", {})

    @classmethod
    @property
    def projects(cls):
        """Return names of available projects."""
        projects = []
        for name in PROJECT_CONFIGS:
            if any(cls(name)._project_files):
                projects.append(name)
        projects.sort()
        return projects

    @property
    def scales(self):
        """Return a titles dictionary with extra fields."""
        scales = self._config.get("scales", {})
        scales.update(SCALE_OVERRIDES)
        for k, v in scales.items():
            for sk, sv in v.items():
                if sv == "na":
                    scales[k][sk] = None
        return scales

    @property
    def scenarios(self):
        """Return just a list of scenario names."""
        return list(self.files.keys())

    @property
    def titles(self):
        """Return a titles dictionary with extra fields."""
        return self._config.get("titles", {})

    @property
    def units(self):
        """Return a units dictionary with extra fields."""
        provided_units = self._config.get("units", {})
        units = UNITS.copy()
        units.update(provided_units)
        return units

    @property
    def _all_files(self):
        """:obj:`generator`: Generator of raw project files."""
        files = []
        if self.options is not None and "file" in self.options:
            for file in self.options.file:
                yield Path(file).expanduser().resolve()
        else:
            yield from self.directory.rglob("*.csv")

    def _check_required_keys_exist(self):
        """Ensure all required keys are present in config file."""
        missing = [
            req for req in self.REQUIREMENTS if self._config.get(req) is None
        ]

        if missing:
            error_message = (
                f"Config for project {self.project!r} missing the following "
                f"keys: {missing}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

    def _check_valid_project_name(self):
        """Ensure project name is not None."""
        if self.project is None:
            raise ValueError("Project input cannot be None!")

    def _extract_fp_from_config(self, fp_key):
        """Extract the file path from the config dictionary."""
        file_path = self._config.get(fp_key)
        if file_path is not None:
            file_path = Path(file_path).expanduser().resolve()
        return file_path

    @property
    def _project_files(self):
        """:obj:`generator`: Generator of project-related files only."""
        for file in self._all_files:
            scenario = strip_rev_filename_endings(file.name)
            if scenario.endswith(".csv"):
                continue
            yield scenario, file

    def _safe_read(self, fp_key, default_fp=None):
        """Read the data corresponding to the config key."""
        path = self._extract_fp_from_config(fp_key) or default_fp
        return _safe_read_csv(path)

    def _set_config(self):
        """Set the config, raise error is project not found."""
        for name, config in PROJECT_CONFIGS.items():
            if name.lower() == self.project.lower():
                self._config = config

        if self._config is None:
            raise ValueError(
                f"No project with name {self.project!r} found in config "
                f"directory"
            )


class Process:
    """Methods for performing standard post-processing steps on reV outputs."""

    import addfips
    import numpy as np
    import pandas as pd

    af = addfips.AddFIPS()

    def __init__(
        self,
        home=".",
        file_pattern="*_sc.csv",
        files=None,
        pixel_sum_fields=[],
        resolution=90,
    ):
        """Initialize Post_Process object.
        Parameters
        ----------
        home : str
            Path to directory containing reV supply-curve tables.
        file_pattern : str
            Glob pattern to filter files in home directory. Defaults to
            "*_sc.csv" (reV supply curve table pattern) and finds all such
            files in the driectory tree.
        pixel_sum_fields : list
            List of field name representing pixel sum characterizations to be
            converted to area and percent of available area fields.
        resolution : int
            The resolution in meters of the exclusion/characterization raster.
        """
        self.home = Path(home)
        self._files = files
        self.file_pattern = file_pattern
        self.pixel_sum_fields = pixel_sum_fields

    def __repr__(self):
        """Return representation string."""
        msg = f"<Post_Process object: home='{self.home}'>"
        return msg

    def process(self):
        """Run all post-processing steps on all files."""
        self.assign_regions()
        self.assign_classes()

    def assign_area(self, file):
        """Assign area to pixel summed characterizations for a file."""  # <-- Some what inefficient, reading and saving twice but covers all cases
        cols = self._cols(file)
        area_fields = [f"{f}_sq_km" for f in self.pixel_sum_fields]
        pct_fields = [f"{f}_pct" for f in self.pixel_sum_fields]
        target_fields = area_fields + pct_fields
        if any([f not in cols for f in target_fields]):
            for field in self.pixel_sum_fields:
                if field in cols:
                    acol = f"{field}_sq_km"
                    pcol = f"{field}_pct"
                    df = self.pd.read_csv(file, low_memory=False)
                    df[acol] = (df[field] * 90 * 90) / 1_000_000
                    df[pcol] = (df[acol] / df["area_sq_km"]) * 100
                    df.to_csv(file, index=False)

    def assign_areas(self):
        """Assign area to pixel summed characterizations for all files."""
        for file in self.files:
            self.assign_area(file)

    def assign_class(self, file, field="windspeed"):
        """Assign a particular resource class to an sc df."""
        col = f"{field}_class"
        cols = self._cols(file)
        if col not in cols:
            df = self.pd.read_csv(file, low_memory=False)
            rfield = self.resource_field(file, field)
            onmap = RESOURCE_CLASSES[field]["onshore"]
            offmap = RESOURCE_CLASSES[field]["offshore"]

            if "offshore" in cols and "wind" in field and "sub_type" in cols:
                # onshore
                ondf = df[df["offshore"] == 0]
                ondf[col] = df[rfield].apply(self.map_range, range_dict=onmap)

                # offshore
                offdf = df[df["offshore"] == 1]

                # Fixed
                fimap = offmap["fixed"]
                fidf = offdf[offdf["sub_type"] == "fixed"]
                clss = fidf[rfield].apply(self.map_range, range_dict=fimap)

                # Floating
                flmap = offmap["floating"]
                fldf = offdf[offdf["sub_type"] == "floating"]
                clss = fldf[rfield].apply(self.map_range, range_dict=flmap)
                fldf[col] = clss

                # Recombine
                offdf = self.pd.concat([fidf, fldf])
                df = self.pd.concat([ondf, offdf])
            else:
                df[col] = df[rfield].apply(self.map_range, range_dict=onmap)
            df.to_csv(file, index=False)

    def assign_classes(self):
        """Assign resource classes if possible to an sc df."""
        for file in self.files:
            for field in RESOURCE_CLASSES.keys():
                self.assign_class(file, field)

    def assign_counties(self):
        """Assign the nearest county FIPS to each point for each file."""
        for file in self.files:
            self.assign_county(file)

    def assign_region(self, file):
        """Assign each point an NREL region."""
        if "nrel_region" not in self._cols(file):
            df = self.pd.read_csv(file)
            df["nrel_region"] = df["state"].map(self.nrel_regions)
            df.to_csv(file, index=False)

    def assign_regions(self):
        """Assign each point an NREL region for each file."""
        for file in self.files:
            self.assign_region(file)

    def map_range(self, x, range_dict):
        """Return class for a given value."""
        for clss, rng in range_dict.items():
            if x > rng[0] and x <= rng[1]:
                return clss

    @property
    def files(self):
        """Return all supply-curve files in home directory."""
        if self._files is None:
            rpattern = f"**/{self.file_pattern}"
            files = sorted(self.home.glob(rpattern))
        else:
            files = self._files
        return files

    @property
    def nrel_regions(self):
        """Return state, NREL region dictionary."""
        regions = {}
        for region, states in REGIONS.items():
            for state in states:
                regions[state] = region
        return regions

    @lru_cache()
    def resource_field(self, file, field="windspeed"):
        """Return the resource field for a data frame."""
        # There is a new situation we have to account for
        if field == "windspeed":
            df = self.pd.read_csv(file, low_memory=False)
            if all([self.np.isnan(v) for v in df["mean_res"]]):
                if "mean_ws_mean-means" in df.columns:
                    field = "mean_ws_mean-means"
                elif "mean_ws_mean" in df.columns:
                    field = "mean_ws_mean"
                else:
                    field = "mean_res"
            else:
                field = "mean_res"
        return field

    def _fips(self, row):
        """Return county FIPS code."""
        return self.af.get_county_fips(row["county"], state=row["state"])

    def _cols(self, file):
        """Return only the columns of a csv file."""
        return self.pd.read_csv(file, index_col=0, nrows=0).columns


@lru_cache(maxsize=16)
def _safe_read_csv(path):
    """Read the csv from path without throwing error if it DNE."""
    try:
        data = pd.read_csv(path)
    except (ValueError, FileNotFoundError):
        data = None
    return data
