# -*- coding: utf-8 -*-
"""Package Data Helpers.

Created on Mon May 23 20:31:32 2022

@author: twillia2
"""
import os

from importlib import resources

import reView


class Paths:
    """Methods for handling paths to package data."""

    @classmethod
    @property
    def home(cls):
        """Return application home directory."""
        return resources.files(reView.__name__).parent

    @classmethod
    @property
    def paths(cls):
        """Return posix path objects for package data items."""
        contents = resources.files(reView.__name__)
        data = [file for file in contents.iterdir() if file.name == "data"][0]
        paths = {}
        for folder in data.iterdir():
            name = os.path.splitext(folder.name)[0].lower()
            paths[name] = folder
        return paths
