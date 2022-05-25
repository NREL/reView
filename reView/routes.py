"""reView routing."""
import logging

from dash.dependencies import Input, Output

from reView.app import app
from reView.pages.scenario import scenario
from reView.pages.config import config
from reView.pages.reeds import reeds
from reView.utils.constants import (
    HOME_PAGE_LOCATION,
    SCENARIO_PAGE_LOCATION,
    CONFIG_PAGE_LOCATION,
    REEDS_PAGE_LOCATION,
)

PAGES = {
    None: scenario.layout,
    HOME_PAGE_LOCATION: scenario.layout,
    SCENARIO_PAGE_LOCATION: scenario.layout,
    CONFIG_PAGE_LOCATION: config.layout,
    REEDS_PAGE_LOCATION: reeds.layout
}


# fmt: off
@app.callback(
    Output("page_content", "children"),
    Input("url", "pathname")
)
def render_page_content(pathname):
    """Output chosen layout from the navigation bar links."""
    logging.getLogger(__name__).info("URL: %s", pathname)
    page = PAGES[pathname]
    return page
