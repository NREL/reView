# -*- coding: utf-8 -*-
"""Config class used for all reView projects.

Created on Sat Aug 15 15:47:40 2020

@author: travis
"""
import logging

from functools import cached_property, lru_cache
from pathlib import Path

import pandas as pd

from reView.utils.constants import COMMON_REV_COLUMN_UNITS, SCALE_OVERRIDES
from reView.utils.functions import (
    deep_replace,
    strip_rev_filename_endings,
    load_project_configs,
)

pd.set_option("mode.chained_assignment", None)

logger = logging.getLogger(__name__)

PROJECT_CONFIGS = load_project_configs()
PROJECT_NAMES = list(PROJECT_CONFIGS.keys())


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
        """:obj:`pandas.DataFrame` or `None`: DataFrame containing
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
        for name in PROJECT_CONFIGS:
            try:
                if any(cls(name)._project_files):
                    yield name
            except ValueError:
                continue

    @classmethod
    @property
    def sorted_projects(cls):
        """Return the sorted names of available projects."""
        return sorted(cls.projects)

    @property
    def scales(self):
        """Return a titles dictionary with extra fields."""
        scales = self._config.get("scales", {})
        scales.update(SCALE_OVERRIDES)
        deep_replace(scales, {"na": None})
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
        units = COMMON_REV_COLUMN_UNITS.copy()
        units.update(provided_units)
        return units

    @property
    def _all_files(self):
        """:obj:`generator`: Generator of raw project files."""
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


@lru_cache(maxsize=16)
def _safe_read_csv(path):
    """Read the csv from path without throwing error if it DNE."""
    try:
        data = pd.read_csv(path)
    except (ValueError, FileNotFoundError):
        data = None
    return data
