# -*- coding: utf-8 -*-
"""Helper classes for all reView projects."""
import os
import inspect
import functools
import logging
from enum import Enum

import addfips
import pandas as pd

from reView import REVIEW_DATA_DIR
from reView.utils.functions import shorten, callback_trigger

logger = logging.getLogger(__name__)


class StrEnum(str, Enum):
    """Enum that acts like a string."""

    def __str__(self):
        return self.value


class DiffUnitOptions(StrEnum):
    """Unit options when difference is selected."""

    ORIGINAL = ".reView.diff.original"
    PERCENTAGE = ".reView.diff.percent"

    @classmethod
    def from_variable_name(cls, name):
        """Parse the input for a `DiffUnitOptions` enum.

        Parameters
        ----------
        name : str
            Input variable name as string. If this string ends in one
            of the `DiffUnitOptions` enumerations, the corresponding
            enum will be returned.

        Returns
        -------
        `DiffUnitOptions` | None
            A `DiffUnitOptions` enum if it was detected in the string,
            `None` otherwise.
        """
        for option in cls:
            if name.endswith(option):
                return option
        return None

    @classmethod
    def remove_from_variable_name(cls, name):
        """Remove any mention of `DiffUnitOptions` enums in the input.

        Parameters
        ----------
        name : str
            Input variable name as string. If this string ends in one
            of the `DiffUnitOptions` enumerations, the ending will be
            stripped, and the resulting string will be returned.

        Returns
        -------
        str
            Input string with suffixes corresponding to
            `DiffUnitOptions` enums removed.
        """
        for ending in cls:
            name = name.removesuffix(ending)
        return name


class FunctionCalls:
    """Class for handling logs and retrieving function arguments."""

    def __init__(self):
        """Initialize FunctionCalls object."""
        self.args = {}

    def __repr__(self):
        """Return FunctionCalls representation string."""
        msg = f"<FunctionCalls: {len(self.args)} function argument dicts>"
        return msg

    def print_all(self):
        """Print all kwargs for each callback in executable fashion."""
        for args in self.args.values():
            for key, arg in args.items():
                print(f"{key}={arg!r}")

    @property
    def all(self):
        """Return executable variable setters for all callbacks.

        The purpose of this function is to compile a string that
        can be used as input to `exec` that sets all the input
        arguments all callbacks as actual variables in your
        namespace. Note that some overlap may occur.

        Returns
        -------
        str
            A string that can be used as input to `exec` that sets all
            the input arguments all callbacks as actual variables in
            your namespace.

        Notes
        -----
        See `FunctionCalls.get` documentation for example(s).
        """
        return "; ".join([f"{key}={arg!r}" for key, arg in self.args.items()])

    def __call__(self, func_name, str_length=None):
        """Return executable variable setters for one callback.

        The purpose of this function is to compile a string that
        can be used as input to `exec` that sets all the input
        arguments to the function `func_name` as actual variables
        in your namespace.

        Parameters
        ----------
        func_name : str
            Name of function to obtain arguments for,
            represented as a string.
        str_length : int, optional
            Option to shorten string to a certain length, or `None` to
            return unaltered string value.

        Returns
        -------
        str
            A string that can be used as input to `exec` that sets all
            the input arguments to the function as actual variables in
            your namespace.

        Examples
        --------
        >>> calls('options_chart_tabs')
        "tab_choice='chart'; chart_choice='cumsum'"
        """
        args = self.args.get(func_name, {})
        args_str = "; ".join([f"{key}={arg!r}" for key, arg in args.items()])
        if str_length:
            args_str = shorten(args_str, str_length)
        return args_str

    def log(self, func):
        """Log the function call.

        Allow extra logging with the `verbose`
        argument.

        Parameters
        ----------
        verbose : bool, optional
            Specify whether to log the function is call itself,
            by default False.
        """

        @functools.wraps(func)
        def _callback_func(*args, **kwargs):
            """Store the arguments used to call the function."""
            name = func.__name__
            sig = inspect.signature(func)
            keys = sig.parameters.keys()

            trigger = callback_trigger()

            self.args[name] = {
                **dict(zip(keys, args)),
                **kwargs,
                "trigger": trigger,
            }

            logger.info("Running %s... (Trigger: %s)", name, trigger)
            logger.debug("Args: %s", self(name, str_length=200))

            return func(*args, **kwargs)

        return _callback_func


class CountyCode:
    """Utility class to calculate county-level codes."""

    _ADD_FIPS = addfips.AddFIPS()
    _COUNTY_EPSG = pd.read_csv(
        os.path.join(REVIEW_DATA_DIR, "county_fp.csv"), dtype=str
    )

    @classmethod
    def fips(cls, county, state):
        """Get FIPS code for county.

        Parameters
        ----------
        county : str
            Name of county.
        state : str
            Name of state.

        Returns
        -------
        str
            County FIPS code.
        """
        return cls._ADD_FIPS.get_county_fips(county, state=state)

    # pylint: disable=no-member
    @classmethod
    def epsg(cls, county, state):
        """Get EPSG code for county.

        Parameters
        ----------
        county : str
            Name of county.
        state : str
            Name of state.


        Returns
        -------
        str
            County EPSG code.
        """
        fips = cls.fips(county, state)
        mask = cls._COUNTY_EPSG.county_fp == fips
        return cls._COUNTY_EPSG.loc[mask, "epsg"].values[0]
