# -*- coding: utf-8 -*-
"""Add fields to output CSVs.

NREL Regions, Configuration variables, etc

Created on Thu Jan  7 07:17:36 2021

@author: twillia2
"""

import pandas as pd
import pathos.multiprocessing as mp
from tqdm import tqdm

from reView.utils.config import Config
from reView.utils.constants import RESOURCE_CLASSES


CONFIG = Config("ATB Onshore - FY21")
CONFIG_PATH = CONFIG.directory
FILES = sorted(CONFIG_PATH.glob("*.csv"))
REGIONS = {
    "Pacific": ["Oregon", "Washington"],
    "Mountain": ["Colorado", "Idaho", "Montana", "Wyoming"],
    "Great Plains": [
        "Iowa",
        "Kansas",
        "Missouri",
        "Minnesota",
        "Nebraska",
        "North Dakota",
        "South Dakota",
    ],
    "Great Lakes": ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin"],
    "Northeast": [
        "Connecticut",
        "New Jersey",
        "New York",
        "Maine",
        "New Hampshire",
        "Massachusetts",
        "Pennsylvania",
        "Rhode Island",
        "Vermont",
    ],
    "California": ["California"],
    "Southwest": ["Arizona", "Nevada", "New Mexico", "Utah"],
    "South Central": ["Arkansas", "Louisiana", "Oklahoma", "Texas"],
    "Southeast": [
        "Alabama",
        "Delaware",
        "District of Columbia",
        "Florida",
        "Georgia",
        "Kentucky",
        "Maryland",
        "Mississippi",
        "North Carolina",
        "South Carolina",
        "Tennessee",
        "Virginia",
        "West Virginia",
    ],
}
NEEDED_SAM_BITS = [
    "mean_fixed_charge_rate",
    "mean_capital_cost",
    "mean_system_capacity",
    "mean_fixed_operating_cost",
]


def reshape_regions():
    """Convert region-states dictionary to state-region dictionary."""
    regions = {}
    for region, states in REGIONS.items():
        for state in states:
            regions[state] = region
    return regions


def capex(df):
    """Recalculate capital costs if needed input columns are present."""
    capacity = df["capacity"]
    capacity_kw = capacity * 1000

    fcr = df["mean_fixed_charge_rate"]

    unit_om = df["mean_fixed_operating_cost"] / df["mean_system_capacity"]
    om = unit_om * capacity_kw

    mean_cf = df["mean_cf"]
    lcoe = df["mean_lcoe"]
    if "raw_lcoe" in df:
        raw_lcoe = df["raw_lcoe"]
    else:
        raw_lcoe = lcoe.copy()

    cc = (
        (lcoe * (capacity * mean_cf * 8760)) - om
    ) / fcr  # Watch out for economies of scale here
    unit_cc = cc / capacity_kw  # $/kw

    raw_cc = (
        (raw_lcoe * (capacity * mean_cf * 8760)) - om
    ) / fcr  # Watch out for economies of scale here
    raw_unit_cc = raw_cc / capacity_kw  # $/kw

    df["capex"] = cc
    df["unit_capex"] = unit_cc
    df["raw_capex"] = raw_cc
    df["raw_unit_capex"] = raw_unit_cc

    return df


def map_range(x, range_dict):
    """Assign a key to x given a list of key, value ranges."""
    keys = []
    for key, values in range_dict.items():
        if x >= values[0] and x < values[1]:
            keys.append(key)
    key = keys[0]
    return key


# pylint: disable=no-member,unsubscriptable-object
def set_field(path, field):
    """Assign a particular resource class to an sc df."""
    df = pd.read_csv(path, low_memory=False)
    col = f"{field}_class"
    if field == "windspeed":  # How can we distinguish solar from wind?
        if "mean_ws_mean-means" in df.columns:
            dfield = "mean_ws_mean-means"
        else:
            dfield = "mean_res"
    else:
        dfield = field

    if col not in df.columns:
        onmap = RESOURCE_CLASSES[field]["onshore"]
        offmap = RESOURCE_CLASSES[field]["offshore"]
        if dfield in df.columns:
            if "offshore" in df.columns and dfield == "windspeed":
                # onshore
                ondf = df[df["offshore"] == 0]
                clss = df[dfield].apply(map_range, range_dict=onmap)
                ondf[col] = clss

                # offshore
                offdf = df[df["offshore"] == 1]

                # Fixed
                fimap = offmap["fixed"]
                fidf = offdf[offdf["sub_type"] == "fixed"]
                clss = fidf[dfield].apply(map_range, range_dict=fimap)

                # Floating
                flmap = offmap["floating"]
                fldf = offdf[offdf["sub_type"] == "floating"]
                clss = fldf[dfield].apply(map_range, range_dict=flmap)
                fldf[col] = clss

                # Recombine
                offdf = pd.concat([fidf, fldf])
                df = pd.concat([ondf, offdf])
            else:
                clss = df[dfield].apply(map_range, range_dict=onmap)
                df[col] = clss
        df.to_csv(path, index=False)
    return df


def set_fields(path):
    """Assign resource classes if possible to an sc df."""
    for field in RESOURCE_CLASSES.keys():
        set_field(path, field)


def update_file(path):
    """Add fields to a single file."""

    # Get scenario from path
    scenario = path.name.replace("_sc.csv", "")

    # Get the extra fields from the project configuration
    file_df = CONFIG.data
    groups = [c for c in file_df.columns if c not in ["name", "file"]]
    entry = file_df[file_df["name"] == path.name]
    entry = entry[groups].to_dict("list")

    # Now read in the data frame and append these values
    df = pd.read_csv(path)
    for key, value in entry.items():
        value = value[0]
        df[key] = value

    # Now add NREL regions
    regions = reshape_regions()
    df["nrel_region"] = df["state"].map(regions)

    # And the scenario
    df["scenario"] = scenario

    df = capex(df)

    for c in df.columns:
        if "Unnamed" in c:
            del df[c]

    df.to_csv(path, index=False)


def update_files():
    """Update all files with added field."""
    with mp.Pool(mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap(update_file, FILES), total=len(FILES)):
            pass


if __name__ == "__main__":
    update_files()
