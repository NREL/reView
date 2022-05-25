#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Sun Mar  8 16:58:59 2020.

@author: travis
"""
import os

from setuptools import setup


REPO_DIR = os.path.abspath(os.path.dirname(__file__))
DESCRIPTION = ("A data portal for reviewing Renewable Energy Potential Model "
               "(reV) outputs")

with open(os.path.join(REPO_DIR, "README.md"), encoding="utf-8") as f:
    README = f.read()

with open("requirements.txt") as f:
    INSTALL_REQUIREMENTS = f.readlines()


DEV_REQUIREMENTS = ["black", "pylint"]
TEST_REQUIREMENTS = ["pytest", "pytest-cov"]
GUNICORN_REQUIREMENTS = ["gunicorn"]


setup(
    name="reView",
    version="0.1.0",
    description=DESCRIPTION,
    long_description=README,
    author="Travis Williams",
    author_email="Travis.Williams@nrel.gov",
    packages=["reView"],
    entry_points={
        "console_scripts": [
            "reView=reView.index:main"
        ],
    },
    zip_safe=False,
    keywords="reView",
    python_requires='>=3.9',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Framework :: Dash",
    ],
    test_suite="tests",
    include_package_data=True,
    package_data={
        "data": [
            "*"
        ]
    },
    install_requires=INSTALL_REQUIREMENTS,
    extras_require={
        "gunicorn": GUNICORN_REQUIREMENTS,
        "test": TEST_REQUIREMENTS,
        "dev": TEST_REQUIREMENTS + DEV_REQUIREMENTS,
    },
)
