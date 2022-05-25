# -*- coding: utf-8 -*-
"""Transition reView project index file."""
import reView.pages.scenario.controller.callbacks  # pylint: disable=unused-import
import reView.pages.reeds.controller.callbacks  # pylint: disable=unused-import

from reView.app import app, server  # pylint: disable=unused-import
from reView.environment.settings import (
    APP_HOST,
    APP_PORT,
    DASH_DEBUG,
    LOG_LEVEL,
)
from reView.routes import render_page_content  # pylint: disable=unused-import
from reView.utils.log import init_logger, log_versions


def main():
    """Run reView."""
    init_logger(level=LOG_LEVEL)
    log_versions()
    app.run_server(
        host=APP_HOST,
        port=APP_PORT,
        debug=DASH_DEBUG
    )

if __name__ == "__main__":
    main()
