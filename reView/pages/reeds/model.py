# -*- coding: utf-8 -*-
"""ReEDS Buildout page data functions.

Created on Mon May 23 21:48:04 2022

@author: twillia2
"""
import pandas as pd

from reView.app import cache4

pd.set_option("mode.chained_assignment", None)


@cache4.memoize()
def cache_reeds(path, year):
    """Create table of single year buildout."""
    df = pd.read_csv(path)
    if year not in df["year"].values:
        df = df[df["year"] == year -1]
    else:
        df = df[df["year"] == year]
    return df
