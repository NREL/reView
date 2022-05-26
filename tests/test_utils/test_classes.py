# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
"""Util classes tests."""
import sys
from io import StringIO

import pytest

from reView.utils.classes import FunctionCalls, DiffUnitOptions

FUNCTION_CALLS = FunctionCalls()


@pytest.fixture(autouse=True)
def cleanup():
    """Reset FUNCTION_CALLS dict after each test."""
    yield
    FUNCTION_CALLS.args = {}


@pytest.fixture
def add_func():
    """Simple add function."""
    @FUNCTION_CALLS.log
    def add(left, right, other=3):
        """Add the inputs."""
        return left + right + other

    return add


@pytest.fixture
def mult_func():
    """Simple add function."""
    @FUNCTION_CALLS.log
    def multiply(x_1, x_2, x_3=1):
        """Add the inputs."""
        return x_1 * x_2 * x_3

    return multiply


class CapturePrint:
    """Context manager to capture print output.

    References
    ----------
    https://tinyurl.com/3u79ad4u
    """

    def __init__(self):
        self.storage = []
        self._stdout = self._string_io = None

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._string_io = StringIO()
        return self.storage

    def __exit__(self, *args):
        self.storage.extend(self._string_io.getvalue().splitlines())
        del self._string_io    # free up some memory
        sys.stdout = self._stdout


def test_function_calls_log(add_func, mult_func):
    """Test `FunctionCalls.log` method."""

    _ = add_func(1, 2)

    assert FUNCTION_CALLS.args == {
        'add': {'left': 1, 'right': 2, 'trigger': 'Unknown'}
    }

    _ = mult_func(5, 6)

    assert len(FUNCTION_CALLS.args) == 2
    assert 'add' in FUNCTION_CALLS.args
    assert 'multiply' in FUNCTION_CALLS.args
    assert FUNCTION_CALLS.args['add'] == {
        'left': 1, 'right': 2, 'trigger': 'Unknown'
    }
    assert FUNCTION_CALLS.args['multiply'] == {
        'x_1': 5, 'x_2': 6, 'trigger': 'Unknown'
    }

    _ = add_func(10, 20)

    assert len(FUNCTION_CALLS.args) == 2
    assert 'add' in FUNCTION_CALLS.args
    assert 'multiply' in FUNCTION_CALLS.args
    assert FUNCTION_CALLS.args['add'] == {
        'left': 10, 'right': 20, 'trigger': 'Unknown'
    }
    assert FUNCTION_CALLS.args['multiply'] == {
        'x_1': 5, 'x_2': 6, 'trigger': 'Unknown'
    }


# pylint: disable=exec-used,undefined-variable
def test_function_calls_call(add_func):
    """Test `FunctionCalls.__call__` method."""

    _ = add_func(1, 2)

    expected = "left=1; right=2; trigger='Unknown'"
    assert FUNCTION_CALLS('add') == expected
    assert FUNCTION_CALLS('multiply') == ''


    assert 'left' not in globals()
    assert 'right' not in globals()
    assert 'trigger' not in globals()

    exec(
        "global left; global right; global trigger;"
        + FUNCTION_CALLS('add')
    )

    assert 'left' in globals()
    assert 'right' in globals()
    assert 'trigger' in globals()

    # pyright: reportUndefinedVariable=false
    assert left  == 1
    assert right == 2
    assert trigger == "Unknown"


def test_function_calls_all(add_func, mult_func):
    """Test `FunctionCalls.all` method."""

    _ = add_func(1, 2)
    _ = mult_func(5, 6)

    out = FUNCTION_CALLS.all.split('; ')
    assert "add={'left': 1, 'right': 2, 'trigger': 'Unknown'}" in out
    assert "multiply={'x_1': 5, 'x_2': 6, 'trigger': 'Unknown'}" in out

    _ = add_func(10, 20)

    out = FUNCTION_CALLS.all.split('; ')
    assert "add={'left': 10, 'right': 20, 'trigger': 'Unknown'}" in out
    assert "multiply={'x_1': 5, 'x_2': 6, 'trigger': 'Unknown'}" in out



def test_function_calls_print_all(add_func, mult_func):
    """Test `FunctionCalls.print_all` method."""

    _ = add_func(1, 2)
    _ = mult_func(5, 6)

    with CapturePrint() as output:
        FUNCTION_CALLS.print_all()

    output = set(output)

    all_expected = [
        'left=1', 'right=2', 'x_1=5', 'x_2=6', "trigger='Unknown'"
    ]
    for expected in all_expected:
        assert expected in output


def test_expected_enums_exist():
    """Test that specific expected enum values exist."""

    assert "ORIGINAL" in dir(DiffUnitOptions)
    assert "PERCENTAGE" in dir(DiffUnitOptions)


def test_difference_unit_options_from_variable_name():
    """Test that specific expected enum values exist."""

    var_with_original_option = f"a_var{DiffUnitOptions.ORIGINAL}"
    assert (DiffUnitOptions.from_variable_name(var_with_original_option)
            == DiffUnitOptions.ORIGINAL)

    var_with_original_option = f"a_var{DiffUnitOptions.PERCENTAGE}"
    assert (DiffUnitOptions.from_variable_name(var_with_original_option)
            == DiffUnitOptions.PERCENTAGE)


def test_difference_unit_options_remove_from_variable_name():
    """Test that specific expected enum values exist."""

    var_with_original_option = f"a_var{DiffUnitOptions.ORIGINAL}"
    assert (DiffUnitOptions.remove_from_variable_name(var_with_original_option)
            == "a_var")

    var_with_original_option = f"another_var{DiffUnitOptions.PERCENTAGE}"
    assert (DiffUnitOptions.remove_from_variable_name(var_with_original_option)
            == "another_var")
