# -*- coding: utf-8 -*-
"""Explore the solar plants in the LBNL dataset.

Author: twillia2
Date: Fri May 26 10:24:29 MDT 2023
"""
import geopandas as gpd

from reView.utils.functions import get_sheet, data_paths, to_geo


HOME = data_paths()["existing_plants"]
GPKG = HOME.joinpath("solar/upv_all_plants_solar.gpkg")
XLSM = HOME.joinpath("solar/2022_utility-scale_solar_data_update.xlsm")


def main():
    """See about layouts capacities and densities, etc."""
    if not GPKG.exists():
        df = get_sheet(XLSM, "Individual_Project_Data")
        df["Solar COD"] = df["Solar COD"].astype(str)
        to_geo(df, GPKG, "upv_all_plants_solar")
    df = gpd.read_file(GPKG)
