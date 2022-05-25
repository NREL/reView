# -*- coding: utf-8 -*-
"""Created on Sun Mar  8 16:58:59 2020.

Heavily based on
https://github.com/CzakoZoltan08/dash-clean-architecture-template
"""

import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), os.getenv("REVIEW_ENV_FILE", ".env"))
load_dotenv(dotenv_path=dotenv_path, override=True)

APP_HOST = os.getenv("HOST")
APP_PORT = int(os.getenv("PORT"))
IS_DEV_ENV = os.getenv("IS_DEV_ENV") == "True"
DASH_DEBUG = os.getenv("DASH_DEBUG", "False") == "True"
LOG_LEVEL = os.getenv("LOG_LEVEL", "NOTSET")
