"""Reduce file sizes of sample dataset."""
from glob import glob

import pandas as pd


FILES = glob("*parquet")
FILES.sort()
KEEPERS = [
    'sc_point_gid',
    'latitude',
    'longitude',
    'timezone',
    'country',
    'state',
    'county',
    'elevation',
    'gid_counts',
    'n_gids',
    'area_sq_km',
    'turbine_x_coords',
    'turbine_y_coords',
    'capacity',
    'fixed_charge_rate',
    'capital_cost',
    'fixed_operating_cost',
    'mean_cf',
    'mean_lcoe',
    'annual_energy-means',
    'annual_gross_energy-means',
    'bespoke_aep',
    'bespoke_capital_cost',
    'bespoke_fixed_operating_cost',
    'bespoke_objective',
    'included_area_capacity_density',
    'n_turbines',
    'wake_losses-means',
    'winddirection',
    'ws_mean'
 ]


def main():
    """Read in each file, subset for needed columns, overwrite."""
    for file in FILES:
        print(file)
        df = pd.read_parquet(file)
        df = df[KEEPERS]
        df.to_parquet(file, index=False)


if __name__ == "__main__":
    main()
