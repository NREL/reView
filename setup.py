# -*- coding: utf-8 -*-
"""Created on Sun Mar  8 16:58:59 2020.

@author: travis
"""
import os
import re

from setuptools import setup, find_packages


REPO_DIR = os.path.abspath(os.path.dirname(__file__))
VERSION_FILE = os.path.join(REPO_DIR, "reView", "version.py")
DESCRIPTION = (
    "A data portal for reviewing Renewable Energy Potential Model "
    "(reV) outputs. It also has limited functionality for viewing ReEDS-to-reV"
    " outputs from the Regional Energy Deployment System Model."
)

with open(VERSION_FILE, encoding="utf-8") as f:
    VERSION = f.read().split("=")[-1].strip().strip('"').strip("'")

with open(os.path.join(REPO_DIR, "README.md"), encoding="utf-8") as f:
    README = f.read()

with open("requirements.txt") as f:
    INSTALL_REQUIREMENTS = f.readlines()

with open("requirements_dev.txt") as f:
    DEV_REQUIREMENTS = f.readlines()

GUNICORN_REQUIREMENTS = ["gunicorn"]

setup(
    name="reView",
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    author="Travis Williams, Paul Pinchuk, and Mike Gleason",
    author_email="Travis.Williams@nrel.gov",
    packages=find_packages(),
    package_dir={"blmlu": "blmlu"},
    entry_points={
        "console_scripts": [
            "reView=reView.index:main",
            "reView-tools=reView.cli:main",
            "unpack-turbines=reView.cli:unpack_turbines",
            "unpack-characterizations=reView.cli:unpack_characterizations",
            "make-maps=reView.cli:make_maps",
            "map-column=reView.cli:map_column"
        ],
    },
    zip_safe=False,
    keywords="reView",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Framework :: Dash",
    ],
    test_suite="tests",
    include_package_data=True,
    package_data={"": ["data/*", "data/**/*"]},
    install_requires=INSTALL_REQUIREMENTS,
    extras_require={
        "gunicorn": GUNICORN_REQUIREMENTS,
        "dev": DEV_REQUIREMENTS,
    }
)
