"""reView routing."""
import logging

from dash.dependencies import Input, Output

from reView.app import app
from reView.pages.rev import view as scenario_view
from reView.pages.reeds import view as reeds_view
from reView.utils.constants import (
    HOME_PAGE_LOCATION,
    SCENARIO_PAGE_LOCATION,
    CONFIG_PAGE_LOCATION,
    REEDS_PAGE_LOCATION,
)

PAGES = {
    None: scenario_view.layout,
    HOME_PAGE_LOCATION: scenario_view.layout,
    SCENARIO_PAGE_LOCATION: scenario_view.layout,
    REEDS_PAGE_LOCATION: reeds_view.layout
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
