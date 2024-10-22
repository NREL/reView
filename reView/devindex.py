# -*- coding: utf-8 -*-
# pylint: skip-file
"""reView project index file."""
import reView.pages.reeds.controller.callbacks
import reView.pages.rev.controller.callbacks

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


def main():
    """Run reView."""
    init_logger(level=LOG_LEVEL)
    log_versions()
    app.run(
        host=APP_HOST,
        port=APP_PORT,
        debug=True,
        # dev_tools_ui=True,
        # dev_tools_hot_reload=True,
        # dev_tools_hot_reload_interval=5000,
        # dev_tools_hot_reload_max_retry=300
    )


if __name__ == "__main__":
    main()
