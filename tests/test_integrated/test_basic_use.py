# -*- coding: utf-8 -*-
"""reView integration tests.

References
----------
https://dash.plotly.com/testing
"""
import pytest
from reView.app import app


# noqa
# pylint: disable=unused-import,import-outside-toplevel
def test_open_review(dash_duo):
    """Test opening review."""
    # this import is part of the test...
    # users will need to run at least this file to see reView, so we need
    # to make sure its importable at minimum.
    import reView.index

    dash_duo.start_server(app)

    # assert dash_duo.find_element("#rev_link_button").is_displayed()
    # assert dash_duo.find_element("#reeds_link_button").is_displayed()
    # assert dash_duo.find_element("#top-level-navbar").is_displayed()

if __name__ == '__main__':
    pytest.main([__file__, '-s'])
