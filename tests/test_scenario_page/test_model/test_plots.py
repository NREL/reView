# -*- coding: utf-8 -*-
"""Scenario Model tests."""
import pytest

from reView.pages.scenario.controller.element_builders import Plots


@pytest.mark.parametrize("bin_size", [None, 10])
def test_plots_bin_boundaries(bin_size):
    """Test that the bin boundaries are calculated correctly."""

    plotter = Plots(
        project="Hydrogen Minimal",
        datasets={},
        plot_title="A Test Plot",
    )

    bin_boundaries = plotter.bin_boundaries(range(50), bin_size=bin_size)
    widths = bin_boundaries[1:] - bin_boundaries[:-1]
    assert len(bin_boundaries) >= 50 / (bin_size or plotter.DEFAULT_N_BINS)
    assert min(bin_boundaries) <= 0
    assert max(bin_boundaries) >= 50
    assert (widths <= (bin_size or plotter.DEFAULT_N_BINS)).all()


@pytest.mark.parametrize("bin_size", [None, 1])
def test_plots_assign_bins(bin_size):
    """Test that the bin boundaries are assigned correctly."""

    plotter = Plots(
        project="Hydrogen Minimal",
        datasets={},
        plot_title="A Test Plot",
    )

    bin_assignments = plotter.assign_bins(
        range(6), bin_size=bin_size, right=True
    )
    assert len(bin_assignments) == 6
    assert (bin_assignments >= 0).all()
    assert (bin_assignments <= 5).all()
    assert len(set(bin_assignments)) == len(bin_assignments)

    bin_assignments = plotter.assign_bins(
        range(6), bin_size=bin_size, right=False
    )
    assert len(bin_assignments) == 6
    assert (bin_assignments >= 0).all()
    assert bin_assignments[-1] >= 5
    assert len(set(bin_assignments)) == len(bin_assignments)
