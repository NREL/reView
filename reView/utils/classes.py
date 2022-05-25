# -*- coding: utf-8 -*-
"""Helper classes for all reView projects."""
import inspect
import functools
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class StrEnum(str, Enum):
    """Enum that acts like a string."""

    def __str__(self):
        return self.value



class Units(StrEnum):
    """Known units and their str representation."""

    CATEGORY = 'category'
    KILOMETERS = 'km'
    LCOE = '$/MWh'
    MEGAWATTS = 'MW'
    METERS = 'm'
    METERS_PER_SECOND = "m/s"
    MILES = 'miles'
    NONE = ''
    PERCENT = '%'
    SQUARE_METERS = 'square km'


class DiffUnitOptions(StrEnum):
    """Unit options when difference is selected."""

    ORIGINAL = '.reView.diff.original'
    PERCENTAGE = '.reView.diff.percent'

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

    def get_all_args(self):
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
        See `FunctionCalls.get_args` documentation for example(s).
        """
        return "; ".join([f"{key}={arg!r}" for key, arg in self.args.items()])

    def get_args(self, func_name):
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

        Returns
        -------
        str
            A string that can be used as input to `exec` that sets all
            the input arguments to the function as actual variables in
            your namespace.

        Examples
        --------
        >>> FUNCTION_CALLS.get_args('options_chart_tabs')
        "tab_choice='chart'; chart_choice='cumsum'"
        """
        args = self.args.get(func_name, {})
        return "; ".join([f"{key}={arg!r}" for key, arg in args.items()])

    def log(self, verbose=False):
        """Log the function call.

        Allow extra logging with the `verbose`
        argument.

        Parameters
        ----------
        verbose : bool, optional
            Specify whether to log the function is call itself,
            by default False.
        """

        def _decorate(func):
            """Decorate the function to log call."""

            @functools.wraps(func)
            def _callback_func(*args, **kwargs):
                """Store the arguments used to call the function."""
                name = func.__name__
                if verbose:
                    logger.info("Running %s...", name)
                else:
                    logger.debug("Running %s...", name)
                sig = inspect.signature(func)
                keys = sig.parameters.keys()
                self.args[name] = {**dict(zip(keys, args)), **kwargs}

                return func(*args, **kwargs)

            return _callback_func

        return _decorate


class Args():
    """Class for handling retrieving function arguments."""

    def __init__(self):
        """Initialize Logger object."""
        self.args = {}

    def __repr__(self):
        """Return Logger representation string."""
        attrs = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()
                           if k != "args"])
        n = len(self.args.keys())
        msg = f"<Logger: {attrs} {n} function argument dicts>"
        return msg

    def printall(self):
        """Print all kwargs for each callback in executable fashion."""
        for func, args in self.args.items():
            for key, arg in args.items():
                if isinstance(arg, str):
                    print(f"{key}='{arg}'")
                else:
                    print(f"{key}={arg}")

    @property
    def getall(self):
        """Return executable variable setters for all callbacks."""
        set_pairs = []
        for func, args in self.args.items():
            for key, arg in args.items():
                if isinstance(arg, str):
                    set_pairs.append(f"{key}='{arg}'")
                else:
                    set_pairs.append(f"{key}={arg}")
        cmd = "; ".join(set_pairs)
        return cmd

    def getargs(self, func_name):
        """Return executable variable setters for one callback."""
        set_pairs = []
        args = self.args[func_name]
        for key, arg in args.items():
            if isinstance(arg, str):
                set_pairs.append(f"{key}='{arg}'")
            else:
                set_pairs.append(f"{key}={arg}")
        cmd = "; ".join(set_pairs)
        return cmd

    def setargs(self, **kwargs):
        """Log the most recent arguments of a callback."""
        caller = inspect.stack()[1][3]
        print(f"Running {caller}...")
        self.args[caller] = kwargs
