# -*- coding: utf-8 -*-
"""Config tests."""
from pathlib import Path

import pytest

from reView import TEST_DATA_DIR
from reView.utils.config import Config


# pylint: disable=protected-access
@pytest.fixture(autouse=True)
def resolve_test_data_dir():
    """Replace TEST_DATA_DIR with path to the test data dir."""

    def patched_extract_fp_from_config(self, fp_key):
        """Extract the file path from the config dictionary."""
        file_path = self._config.get(fp_key)
        if file_path is not None:
            file_path = file_path.replace("TEST_DATA_DIR", TEST_DATA_DIR)
            file_path = Path(file_path).expanduser().resolve()
        return file_path

    Config._extract_fp_from_config = patched_extract_fp_from_config


# pylint: disable=protected-access
@pytest.fixture(autouse=True)
def reset_configs():
    """Reset singleton config info."""
    Config._all_configs = {}
    yield


def test_invalid_project_input():
    """Test invalid project name input (None)"""

    with pytest.raises(ValueError) as excinfo:
        Config(None)

    assert "Project input cannot be None!" in str(excinfo.value)


def test_dne_project_name():
    """Test project name that does not exist."""

    with pytest.raises(ValueError) as excinfo:
        Config("Project Name DNE")

    assert "No project with name 'Project Name DNE'" in str(excinfo.value)


def test_project_no_directory():
    """Test that error si thrown for project with no directory key."""

    with pytest.raises(ValueError) as excinfo:
        Config("Test No Dir")

    assert "missing the following keys" in str(excinfo.value)


def test_config_path_resolves_correctly():
    """Test that relative directory is resolved correctly."""

    config = Config("Hydrogen Relative")
    correct_path = (Path.home() / "hydrogen").resolve()

    assert config.directory == correct_path


def test_config_no_var():
    """Test that files are checked if var file is not provided."""

    config = Config("Hydrogen No Var No Demand")

    files_1 = list(config._all_files)
    files_2 = list(config._all_files)
    assert files_1 == files_2  # check that generator is re-initalized

    assert "empty_data_1" in config.files
    assert config.files["empty_data_1"] in files_1
    assert "should_be_excluded.csv" not in files_1

    assert config.demand_data is None


def test_config_no_var_but_default_file_exists():
    """Test that config looks for default var file."""

    config = Config("Hydrogen No Var But With Default")

    assert config.files
    assert isinstance(config.files, dict)


# pylint: disable=use-implicit-booleaness-not-comparison
def test_properties_of_minimal_config():
    """Test default values for minimal config."""

    config = Config("Hydrogen Minimal")

    assert config.directory
    assert config.options is None
    assert config.demand_data is None
    assert config.characterizations_cols == []
    assert config.parameters == {}
    assert config.low_cost_groups == {}
    assert config.groups == {}
    assert config.titles == {}
    assert len(config.scales) >= 0
    assert len(config.units) >= 0
    assert len(config.scenarios) >= 2
    assert len(config.scenarios) >= 2
    assert "empty_data_1" in config.scenarios
    assert "empty_data_2" in config.scenarios

    assert repr(config) == "Config('Hydrogen Minimal')"
    assert "Config" in str(config)
    assert "Hydrogen Minimal" in str(config)


def test_config_is_singleton():
    """Test that config object is singleton."""

    config1 = Config("Hydrogen Minimal")
    config2 = Config("Hydrogen Minimal")

    assert config1 is config2


# pylint: disable=unsupported-membership-test
def test_config_projects():
    """Test config projects property."""
    projects = set(Config.projects)

    assert len(projects) >= 3
    assert "Hydrogen No Var But With Default" in projects
    assert "Hydrogen No Var No Demand" in projects
    assert "Hydrogen Minimal" in projects
