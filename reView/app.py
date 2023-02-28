# -*- coding: utf-8 -*-
"""Create dash application objects, server, and data caches.

Created on Sun Aug 23 16:39:45 2020

@author: travis
"""
from pathlib import Path

import dash
import dash_bootstrap_components as dbc

from flask_caching import Cache

from reView.layout.layout import layout


DATA_DIR = Path("~/.review/cache-directory").expanduser()
DATA_DIR.mkdir(exist_ok=True, parents=True)


app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.layout = layout
server = app.server

# Dash adds a StreamHandler by default, as do we,
# so we get rid of the Dash handler instance in favor of our own
for handler in app.logger.handlers:
    app.logger.removeHandler(handler)

# Create simple cache for storing updated supply curve tables
cache = Cache(
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": DATA_DIR.joinpath("cache"),
        "CACHE_THRESHOLD": 10,
    }
)

# Create another cache for storing filtered supply curve tables
cache2 = Cache(
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": DATA_DIR.joinpath("cache2"),
        "CACHE_THRESHOLD": 10,
    }
)

# Create another cache for storing filtered supply curve tables
cache3 = Cache(
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": DATA_DIR.joinpath("cache3"),
        "CACHE_THRESHOLD": 10,
    }
)

# Cache for reeds build out tables
cache4 = Cache(
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": DATA_DIR.joinpath("cache4"),
        "CACHE_THRESHOLD": 10,
    }
)

# Should we just toss everything in one big cache?

cache.init_app(server)
cache2.init_app(server)
cache3.init_app(server)
cache4.init_app(server)
