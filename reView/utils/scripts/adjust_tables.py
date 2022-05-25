# -*- coding: utf-8 -*-
"""Adjust BLM Leas Rate SC tables.

Created on Thu Jan 13 15:17:50 2022

@author: travis
"""
import os

from glob import glob

import numpy as np
import pandas as pd

from tqdm import tqdm

SOURCE = os.path.expanduser("~/review_datasets/blm_lease_rates")
TRGT = SOURCE + "_adjusted"
KEEPERS = ['sc_gid', 'res_gids', 'gen_gids', 'gid_counts', 'n_gids', 'mean_cf',
           'mean_lcoe', 'mean_res', 'capacity', 'area_sq_km', 'latitude',
           'longitude', 'country', 'state', 'county', 'elevation', 'timezone',
           'cnty_fips', 'lbnl_convex_hull_existing_farms_2018', 'mean_fixed_operating_cost',
           'lbnl_upv_1km_buffer', 'sc_point_gid', 'sc_row_ind', 'sc_col_ind',
           'res_class', 'trans_gid', 'trans_type', 'lcot', 'total_lcoe',
           'dist_km', 'trans_cap_cost_per_mw', 'rate_usd_ac', 'adjusted_om',
           'adjusted_mean_lcoe', 'nrel_region', 'mean_system_capacity',
           'lbnl_convex_hull_existing_farms_2021', 'rate_usd_mw',
           'usa_mrlc_nlcd2011', 'dist_mi', 'trans_capacity', 'trans_cap_cost',
           'transmission_multiplier']

SOLAR = pd.read_csv("~/review_datasets/open_access_sc.csv",
                    usecols=["sc_point_gid", "lbnl_upv_1km_buffer"])
SOLAR["lbnl_upv_1km_buffer"][np.isnan(SOLAR["lbnl_upv_1km_buffer"])] = 0


def adjust(file):
    """Separate lease/capacity rates, calculate total charge."""
    print(file)
    df = pd.read_csv(file)
    df = df[KEEPERS]
    del df["lbnl_upv_1km_buffer"] 
    df = pd.merge(df, SOLAR, on="sc_point_gid")
    if "awea" not in file: 
        capfee = int(os.path.basename(file).split("_")[4].replace("capfee", ""))
        density = int(os.path.basename(file).split("_")[3].replace("acmw", ""))
        option = os.path.basename(file).split("_")[2]
        total_charge = capfee + df["rate_usd_ac"]
        rent = df["rate_usd_ac"]
    else:
        capfee = 0
        density = 0
        rent = 0
        option = "None"
        total_charge = 0

    technology = os.path.basename(file).split("_")[0]
    df["adjusted_om"] = df["adjusted_om"] / df["capacity"]
    df["blm_lease_option"] = option
    df["blm_capacity_fee"] = capfee
    df["blm_rent"] = rent
    df["blm_density"] = density
    df["blm_total_charge"] = total_charge
    df["technology"] = technology
    dst = os.path.join(TRGT, os.path.basename(file))
    df.to_csv(dst, index=False)


def main():
    """Separate lease/capacity rates, calculate total charge for each file."""
    os.makedirs(TRGT, exist_ok=True)
    files = glob(os.path.join(SOURCE, "*csv"))
    for file in files:
        adjust(file)


if __name__ == "__main__":
    main()
