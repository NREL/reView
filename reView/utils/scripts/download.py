# -*- coding: utf-8 -*-
"""
Get sample data for reViewer
"""
import os
import multiprocessing as mp
import sys
import urllib
import xarray as xr

from dask.distributed import Client
from tqdm import tqdm
from gdalmethods import Data_Path


# A single master data folder?
DP = Data_Path("~/data")
os.makedirs(DP.folder_path, exist_ok=True)

# Standardized Evapotranspiration Index - 6 months
URL = "https://wrcc.dri.edu/wwdt/data/PRISM/spei6/"
URLS = [os.path.join(URL, f"spei6_{i}_PRISM.nc") for i in range(1, 13)]


def download(url):
    """Download a single file."""
    nc_file = DP.join(os.path.basename(url))
    if not os.path.exists(nc_file):
        with open(nc_file, "wb") as file:
            try:
                # pylint: disable=consider-using-with
                response = urllib.request.urlopen(url).read()
                file.write(response)
            except urllib.error.URLError as error:
                print(error.reason)
    return nc_file


def main():
    """Download 4 files at a time, combine into single netcdf file."""

    with mp.Pool(4) as pool:
        nc_files = []
        for nc_file in tqdm(pool.imap(download, URLS), total=len(URLS),
                            position=0, file=sys.stdout):
            nc_files.append(nc_file)

    # making a single data set
    spei = xr.open_mfdataset(nc_files, concat_dim="day", combine='nested')
    spei = spei.sortby("day")

    # save to singular netcdf
    with Client():
        spei = spei.compute()
    spei.to_netcdf(DP.join("spei6.nc"))

    # save to tiledb?

    # save to postgres?

    # save to rasdaman?

    # save to some faster, better thing?


if __name__ == "__main__":
    main()
