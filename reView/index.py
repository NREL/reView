# -*- coding: utf-8 -*-
# pylint: skip-file
"""Transition reView project index file."""
import json

import reView.pages.reeds.controller.callbacks
import reView.pages.scenario.controller.callbacks

from reView.paths import Paths
from reView.app import app, server
from reView.environment.settings import (
    APP_HOST,
    APP_PORT,
    DASH_DEBUG,
    LOG_LEVEL,
)
from reView.routes import render_page_content
from reView.utils.log import init_logger, log_versions
from reView.utils import calls


def set_sample_path():
    """Set local path to sample dataset."""
    sample_dir = Paths.paths["samples"]
    config_path = Paths.home.joinpath("configs/sample.json")
    with open(config_path, "r") as file:
        config = json.load(file)
        config["directory"] = sample_dir
    with open(config_path, "w") as file:
        file.write(json.dumps(config, indent=4))


def main():
    """Run reView."""
    set_sample_path()
    init_logger(level=LOG_LEVEL)
    log_versions()
    app.run_server(host=APP_HOST, port=APP_PORT, debug=DASH_DEBUG)


if __name__ == "__main__":
    main()
