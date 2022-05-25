# -*- coding: utf-8 -*-
"""
Conversion functions for reV outputs.

Created on Fri May 29 22:15:54 2020

@author: travis
"""

import datetime as dt
import os

from glob import glob

import click
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from netCDF4 import Dataset
from osgeo import osr
from scipy.spatial import cKDTree
from shapely.geometry import Point


NONDATA = ["meta", "time_index"]

COORDINATE_SYSTEM_AUTHORITIES = {
    "albers": 102008,
    "wgs": 4326
}


def convert_h5(src, dst, res=0.0215, overwrite=True):
    """Covert a point data frame into a 3D gridded dataset.

    Parameters
    ----------
    file : str
        Path to an HDF5 or csv reV output file.


    Sample Arguments:
    ----------------
    src = "/home/travis/github/reViewer/data/ipm_wind_florida/ipm_wind_cfp_fl_2012.h5"
    dst = "/home/travis/github/reViewer/data/netcdfs/ipm_wind_cfp_fl_2012.nc"
    res = 0.018
    """

    # Open dataset and build data frame (watch memory)
    datasets = {}
    with h5py.File(src, "r") as ds:
        keys = list(ds.keys())
        if "meta" in keys:
            crds = pd.DataFrame(ds["meta"][:])
        elif "coordinates" in datasets:
            crds = pd.DataFrame(ds["coordinates"][:])
        elif "latitude" in datasets:  # <-------------------------------------- Search for any of a list of possible lat/lon names
            lats = pd.DataFrame(ds["latitude"][:])
            lons = pd.DataFrame(ds["longitude"][:])
            crds = pd.merge(lats, lons)
        else:
            raise ValueError("No coordinate data found in " + dst)

        crds["gid"] = crds.index
        crds = crds[["latitude", "longitude", "gid"]]
        time = [t.decode() for t in ds["time_index"][:]]
        keys = [k for k in ds.keys() if k not in NONDATA]
        for k in keys:

            # Read as Pandas Data Frame (easy, high memory)
            df = pd.DataFrame(ds[k][:])
            df.columns = crds["gid"].values
            df.index = time
            df = df.T
            df = crds[["latitude", "longitude"]].join(df)  
            datasets[k] = df

    # Create geo data frame so we can reproject
    df = to_geo(df)

    # Create a normal grid and assign points with nearest neighbors
    array, transform, time = to_grid(df, res=res)

    # Create an NC dataset out of this
    to_netcdf(array, dst, time, transform, attributes=None)


def resamples(src, n=6):
    """Resample a gridded dataset to n resolutions. I'm not sure we'll be able
    to use the GDAL utilities for this....examine ways to do internally.

    Parameterse
    ----------
    src : str
        Path to an HDF5 or csv reV output file.
    n : int
        The number of different resolutions in which to resample the file.

    Returns
    -------
    list
        A list of paths to output files.
    
    Sample Arguments
    ----------------
    src = "/home/travis/github/reViewer/data/ipm_wind_florida/ipm_wind_cfp_fl_2012.nc"
    n = 6
    """

    # Expand path
    src = os.path.expanduser(src)

    # Open highest resolution original file    
    ds = xr.open_dataset(src)

    # Build regridder
    x = ds["lon"].values
    y = ds["lat"].values
    res = np.diff(x)[0]

    # The spatial reference needs to be removed, and the transform replaced
    crs = ds["spatial_ref"].copy()
    del ds["spatial_ref"]

    scales = {}
    for scale in np.arange(1, n + 1, 1):
        scales[scale] = res * scale

    for scale, res in scales.items():

        # New file name
        zoom_level = n + 1 - scale
        dst = src.replace(".nc", "_{}.nc".format(zoom_level))

        # Create Target Grid
        grid = xr.Dataset({"lat": (["lat"], np.arange(y[0], y[-1], res)),
                           "lon": (["lon"], np.arange(x[0], x[-1], res))})

        # Regrid
        regridder = xe.Regridder(ds, grid, 'bilinear')
        rds = regridder(ds)

        # Compression on each dataset before adding spatial reference back in
        encoding = {k: {'zlib': True} for k in rds.keys()}

        # Rewrite geotransform
        crs.attrs["GeoTransform"] = [res, 0, x[0], 0, res, y[0]]
        rds = rds.assign(spatial_ref = crs)

        # Save
        rds.to_netcdf(dst, encoding=encoding)

    # Remove all of the weight files
    weights = glob("bilinear*")
    for w in weights:
        os.remove(w)


def to_geo(df):
    """
    Create a GeoDataFrame out of a Pandas Data Frame with coordinates.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Pandas data frame with lat/lon coordinates.

    Returns
    -------
    gdf:
        geopandas.
    """

    # Coordinate could be any of these
    lons = ["longitude", "lon", "long",  "x"]
    lats = ["latitude", "lat", "y"]

    # Make sure everything is lower case
    df.columns = [c.lower() for c in df.columns]
    df.columns = [c if c not in lons else "lon" for c in df.columns]
    df.columns = [c if c not in lats else "lat" for c in df.columns]

    # Now make a point out of each row and creat the geodataframe
    df["geometry"] = df.apply(to_point, axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="epsg:4326")

    return gdf


def to_grid(df, res):
    """
    Convert coordinates from an irregular point dataset into an even grid.

    Parameters
    ----------
    df : pandas.core.Frame.DataFrame
        A pandas data frame with x/y coordinates.
    res: int | float
        The resolution of the target grid.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Returns a 3D array (y, x, time) of data values a 2D array of coordinate
        values (nxy, 2).


    Notes
    -----
    - This only takes about a minute for a ~500 X 500 X 8760 dim dataset, but
    it eats up memory. If we saved the df to file, opened it as a dask data
    frame, and generated the arrays as dask arrays we might be able to save
    space.

    - At the moment it is a little awkardly shaped, just because I haven't
    gotten to it yet. 
    """

    # At the end of all this the actual data will be inbetween these columns
    non_values = ["lat", "lon", "y", "x", "geometry", "gx", "gy", "ix", "iy"]

    # Get the extent
    df["lon"] = df["geometry"].apply(lambda x: x.x)
    df["lat"] = df["geometry"].apply(lambda x: x.y)
    minx = df["lon"].min()
    miny = df["lat"].min()
    maxx = df["lon"].max()
    maxy = df["lat"].max()

    # Estimate target grid coordinates
    gridx = np.arange(minx, maxx + res, res)
    gridy = np.arange(miny, maxy + res, res)
    grid_points = np.array(np.meshgrid(gridy, gridx)).T.reshape(-1, 2)

    # Go ahead and make the geotransform 
    geotransform = [res, 0, minx, 0, res, miny]

    # Get source point coordinates
    pdf = df[["lat", "lon"]]
    points = pdf.values

    # Build kdtree
    ktree = cKDTree(grid_points)
    dist, indices = ktree.query(points)

    # Those indices associate grid point coordinates with the original points
    df["gx"] = grid_points[indices, 1]
    df["gy"] = grid_points[indices, 0]

    # And these indices indicate the 2D cartesion positions of the grid
    df["ix"] = df["gx"].apply(lambda x: np.where(gridx == x)[0][0])
    df["iy"] = df["gy"].apply(lambda x: np.where(gridy == x)[0][0])

    # Now we want just the values from the data frame, no coordinates
    value_cols = [c for c in df.columns if c not in non_values]
    vdf = df[value_cols]
    time = vdf.columns
    values = vdf.T.values  # Putting time first here

    # Okay, now use this to create our 3D empty target grid
    grid = np.zeros((values.shape[0], gridy.shape[0], gridx.shape[0]))

    # Now, use the cartesian indices to add the values to the new grid
    grid[:, df["iy"].values, df["ix"].values] = values # <--------------------- Check these values against the original dataset

    # Holy cow, did that work?
    return grid, geotransform, time


def to_netcdf(array, dst, time, transform, attributes=None, clobber=True):
    """
    Convert a 3D numpy time series dataset into a netcdf dataset.

    Parameters
    ----------
    array : np.ndarray
        Numpy array of vairable data.
    dst : str
        Target file path.
    time : list-like
        A list of datetime compatiable time strings.
    transform : list-like
        GDAL formatted geotransform.
    proj : str | int 
        A proj4 string or EPSG code that corresponds with the arrays coordinate
        reference system.
    attributes : dict
        A dictionary with attribute names and values.

    Notes
    -----
    - This only works for Alber's Equal Area Conic North America at the moment
    """

    # For attributes
    todays_date = dt.datetime.today()
    today = np.datetime64(todays_date)

    # Reformat time strings
    def to_hours(time):
        time_fmt = "%Y-%m-%d %H:%M:%S"
        trgt_fmt = "%Y-%m-%d %H:%M"
        base = dt.datetime.strptime(time[0], time_fmt)
        time_units = "hours since {}".format(dt.datetime.strftime(base,
                                                                  trgt_fmt))
        times = [dt.datetime.strptime(t, time_fmt) for t in time]
        times = [dt.datetime.strftime(t, trgt_fmt) for t in times]
        times = [dt.datetime.strptime(t, trgt_fmt) for t in times]
        time_deltas = [t - base for t in times]
        hours = []
        for d in time_deltas:
            days, seconds = d.days, d.seconds
            hour = days * 24 + seconds // 3600
            hours.append(hour)

        return hours, time_units

    hours, time_units = to_hours(time)

    # Get spatial information
    if len(array.shape) == 3:
        ntime, nlat, nlon = np.shape(array)
    else:
        nlat, nlon = np.shape(array)

    # Create Coordinates (res, 0, minx, 0, res, miny)
    lons = np.arange(nlon) * transform[0] + transform[2]
    lats = np.arange(nlat) * transform[4] + transform[5]

    # Create Dataset
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    nco = Dataset(dst, mode="w", format='NETCDF4', clobber=clobber)

    # Time
    nco.createDimension('time', None)
    times = nco.createVariable('time', 'f8', ('time',))
    times.units = time_units
    times.standard_name = 'time'
    times.calendar = 'gregorian'

    # Latitude
    nco.createDimension('lat', nlat)
    latitudes = nco.createVariable('lat',  'f4', ('lat',))
    latitudes.units = 'degrees_north'
    latitudes.standard_name = 'latitude'

    # Longitude
    nco.createDimension('lon', nlon)
    longitudes = nco.createVariable('lon',  'f4', ('lon',))
    longitudes.units = 'degrees_east'
    longitudes.standard_name = 'longitude' 

    # Variables (for var, attrs in variables.items():)
    variable = nco.createVariable('value', 'i2', ('time', 'lat', 'lon'),
                                  fill_value=-9999, zlib=True)
    variable.standard_name = 'index'
    variable.units = 'unitless'
    variable.long_name = 'Index Value'
    variable.scale_factor = 1
    variable.setncattr('grid_mapping', 'spatial_ref')

    # Mean Variables 
    mean_variable = nco.createVariable('value_mean', 'i2', ('lat', 'lon'),
                                       fill_value=-9999, zlib=True)
    mean_variable.standard_name = 'index'
    mean_variable.units = 'unitless'
    mean_variable.long_name = 'Index Value'
    mean_variable.scale_factor = 1
    mean_variable.setncattr('grid_mapping', 'spatial_ref')

    # Appending the CRS information
    refs = osr.SpatialReference()
    refs.ImportFromEPSG(4326)
    wkt = refs.ExportToWkt()

    crs = nco.createVariable('spatial_ref', 'i4')
    crs.spatial_ref = wkt
    crs.GeoTransform = transform
    crs.grid_mapping_name = 'latitude_longitude'
    crs.long_name = 'Lon/Lat WGS 84'
    crs.geographic_crs_name = 'WGS 84'  # is this buried in refs anywhere?
    crs.longitude_of_prime_meridian = 0.0
    crs.semi_major_axis = refs.GetSemiMajor()
    crs.inverse_flattening = refs.GetInvFlattening()

    # Global Attributes
    # nco.title = ""
    # nco.subtitle = ""
    # nco.description = ""
    # nco.original_author = ''
    nco.date = pd.to_datetime(str(today)).strftime('%Y-%m-%d')
    # nco.citation = ""
    nco.Conventions = "CF-1.6"

    # Write
    latitudes[:] = lats
    longitudes[:] = lons
    times[:] = hours
    variable[:] = array
    marray = np.nanmean(array, axis=0)
    mean_variable[:] = marray

    # Done
    nco.close()


def to_point(row):
    """Make a point out of each row of a pandas data frame."""
    return Point((row["lon"], row["lat"]))


# @click.command()
def main(src, outdir="."):
    """Take an HDF5 or csv reV output (or perhaps a resource file), grid it
    into an NC file, take that and resample into 4-5 progressively coarser
    resolutions, and write each to file. It might be more efficient for the
    scatter mapbox to save back to HDF5 format.

    src = "/home/travis/github/reViewer/data/ipm_wind_florida/ipm_wind_cfp_fl_2012.h5"
    outdir = "/home/travis/github/reViewer/data/ipm_wind_florida"
    """

    # Create the target path
    extension = os.path.splitext(src)[1]
    dstfile = os.path.basename(src).replace(extension, ".nc")
    outdir = os.path.expanduser(os.path.abspath(outdir))
    dst = os.path.join(outdir, dstfile)
    print("Converting " + src + "...")

    # And create an evenly gridded projected dataset
    convert_h5(src, dst, overwrite=True)

    # And create 5 or 6 progressively coarser datasets
    resamples(dst, 5)



if __name__ == "__main__":
    src = "/home/travis/github/reViewer/data/ipm_wind_florida/ipm_wind_cfp_fl_2012.h5"
    outdir = "/home/travis/github/reViewer/data/ipm_wind_florida"
    main(src, outdir)
