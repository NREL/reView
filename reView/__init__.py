# -*- coding: utf-8 -*-
"""The reV model viewer (reView)"""
import os

from .paths import Paths
from reView.version import __version__

REVIEW_DIR = os.path.dirname(os.path.realpath(__file__))
REVIEW_CONFIG_DIR = os.path.join(os.path.dirname(REVIEW_DIR), "configs")
TEST_DATA_DIR = os.path.join(os.path.dirname(REVIEW_DIR), "tests", "data")
