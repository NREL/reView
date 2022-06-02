# -*- coding: utf-8 -*-
"""The reV model viewer (reView)"""
import os


from pint import UnitRegistry

from reView.version import __version__

REVIEW_DIR = os.path.dirname(os.path.realpath(__file__))
REVIEW_DATA_DIR = os.path.join(REVIEW_DIR, "data")
REVIEW_CONFIG_DIR = os.path.join(os.path.dirname(REVIEW_DIR), "configs")
TEST_DATA_DIR = os.path.join(os.path.dirname(REVIEW_DIR), "tests", "data")

# May want to move this somewhere in utils
UNITS = UnitRegistry()
Q_ = UNITS.Quantity
UNITS.define("percent = [percent] = % = pct")
UNITS.define("category = 1 = _ = cat")
