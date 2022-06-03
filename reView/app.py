# -*- coding: utf-8 -*-
"""Create dash application objects, server, and data caches.

Created on Sun Aug 23 16:39:45 2020

@author: travis
"""
import json

import dash
import dash_bootstrap_components as dbc

from flask_caching import Cache

from reView.layout.layout import layout
from reView.paths import Paths


app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.layout = layout
server = app.server


# Set local path to sample dataset.
sample_dir = str(Paths.paths["samples"])
config_path = Paths.home.joinpath("configs/sample.json")
with open(config_path, "r") as file:
    config = json.load(file)
    config["directory"] = sample_dir
with open(config_path, "w") as file:
    file.write(json.dumps(config, indent=4))      


# Dash adds a StreamHandler by default, as do we,
# so we get rid of the Dash handler instance in favor of our own
for handler in app.logger.handlers:
    app.logger.removeHandler(handler)

# Create simple cache for storing updated supply curve tables
cache = Cache(
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": "cache-directory/cache",
        "CACHE_THRESHOLD": 10,
    }
)

# Create another cache for storing filtered supply curve tables
cache2 = Cache(
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": "cache-directory/cache2",
        "CACHE_THRESHOLD": 10,
    }
)

# Create another cache for storing filtered supply curve tables
cache3 = Cache(
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": "cache-directory/cache3",
        "CACHE_THRESHOLD": 10,
    }
)

# Cache for reeds build out tables
cache4 = Cache(
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": "cache-directory/cache3",
        "CACHE_THRESHOLD": 10,
    }
)

cache.init_app(server)
cache2.init_app(server)
cache3.init_app(server)
cache4.init_app(server)
