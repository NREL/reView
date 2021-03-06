# -*- coding: utf-8 -*-
"""Fixtures and setup for use across all tests."""
from pathlib import Path

import pytest

from selenium.webdriver.chrome.options import Options

import reView.utils.config

from reView import TEST_DATA_DIR
from reView.utils.functions import load_project_configs


@pytest.fixture
def test_data_dir():
    """Return TEST_DATA_DIR as a `Path` object."""
    return Path(TEST_DATA_DIR)


# pylint: disable=redefined-outer-name
@pytest.fixture
def test_config_dir(test_data_dir):
    """Return test config directory as a `Path` object."""
    return test_data_dir / "configs"


# pylint: disable=redefined-outer-name
@pytest.fixture(autouse=True)
def test_configs(test_config_dir):
    """Load test configs."""
    old_configs = reView.utils.config.PROJECT_CONFIGS
    test_configs_ = load_project_configs(test_config_dir)
    reView.utils.config.PROJECT_CONFIGS = test_configs_

    yield

    reView.utils.config.PROJECT_CONFIGS = old_configs


def pytest_setup_options():
    """Recommended setup based on https://dash.plotly.com/testing."""
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--headless")
    return options
