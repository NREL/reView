# -*- coding: utf-8 -*-
"""Layout generator.

Methods for converting individual turbine coordinates to layouts in plotly and
distributing data appropriately.

Note that this will only work for the US atm since we're using
coordinate reference system lookup made containing only the US.

Created on Wed Apr 13 10:37:14 2022

@author: twillia2
"""
import json
import multiprocessing as mp

from functools import partial
from multiprocessing import cpu_count

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj

from pandarallel import pandarallel
from shapely import geometry
from tqdm import tqdm

pyproj.network.set_network_enabled(False)  # Resolves VPN issues


SPLIT_COLS = ["capacity", "annual_energy-means"]


# pylint: disable=invalid-name,unpacking-non-sequence
class BespokeUnpacker:
    """Methods for manipulating Bespoke reV outputs."""

    def __init__(self, df, clicksel=None, sc_point_gid=None,
                 trgt_crs="esri:102008"):
        """Initialize BespokeUnpacker object.

        Parameters
        ----------
        df : pd.core.frame.DataFrame
            A reV supply curve pandas data frame.
        clicksel : dict
            Dictionary containing plotly point attributes from a
            scattermapbox point selection. Defaults to None.
        sc_point_gid : int
            Supply curve grid id. An ID indicating a specific site within a
            full supply curve grid. If not provided, a clicksel dictionary
            is required. Defaults to None.
        target_crs : str
            CRS of unpacked turbines.
        """
        self.df = df
        self.clicksel = clicksel
        self.sc_point_gid = sc_point_gid
        self.src_crs = "epsg:4326"
        self.trgt_crs = trgt_crs
        if sc_point_gid or clicksel:
            self._declick(clicksel)

    def __repr__(self):
        """Return representation string for Layout object."""
        attrs = ["index", "lat", "lon", "text"]
        attrs = ", ".join([f"{a}={attr}" for a, attr in self.__dict__.items()])
        return f"<BespokeUnpacker object: {attrs}>"

    def get_xy(self, row):
        """Project row to an equal area crs."""
        lat, lon = row["latitude"], row["longitude"]
        transformer = pyproj.Transformer.from_crs(
            self.src_crs,
            self.trgt_crs,
            always_xy=True
        )
        return transformer.transform(lon, lat, errcheck=True)

    @property
    def spacing(self):
        """Infer the spacing between points."""
        # Assuming a 128 agg factor for now
        spacing = 11_520  # Make dynamic
        return spacing

    def to_wgs(self, rdf):
        """Convert x, y coordinates in unpacked dataframe to WGS84."""
        xs = rdf["x"].values
        ys = rdf["y"].values
        transformer = pyproj.Transformer.from_crs(
            self.trgt_crs,
            self.src_crs,
            always_xy=True
        )
        lons, lats = transformer.transform(xs, ys)
        rdf["longitude"] = lons
        rdf["latitude"] = lats
        del rdf["x"]
        del rdf["y"]
        rdf = rdf[self.df.columns]
        return rdf

    def unpack_row(self, row, capacity_col="capacity_ac_mw"):
        """Unpack a single row (pandas Series) in a supply curve table."""
        x, y = self.get_xy(row)

        # Get bottom left coordinates
        blx = x - (self.spacing / 2)
        bly = y - (self.spacing / 2)

        # Get layout
        xs = json.loads(row["turbine_x_coords"])
        ys = json.loads(row["turbine_y_coords"])
        xs = [x + blx for x in xs]
        ys = [y + bly for y in ys]

        # use len(xs) to determine number of turbines because
        # nturbines does not appear to be a standard column
        turbine_capacity_mw = row[capacity_col] / len(xs)

        # Build new data frame entries for each turbine
        new_rows = []
        for i, x in enumerate(xs):
            new_row = row.copy()
            # overwrite existing capacity column (which is typically system
            # capacity in mw) with turbine capacity in kw for this turbine row.
            # This maintains compatibility with how capacity is summed and
            # displayed in the dashboard
            new_row[capacity_col] = turbine_capacity_mw
            new_row["x"] = x
            new_row["y"] = ys[i]
            new_rows.append(new_row)

        return new_rows

    def unpack_turbines(self, drop_sc_points=False,
                        capacity_col="capacity_ac_mw"):
        """Unpack bespoke turbines if possible.

        Parameters
        ----------
        drop_sc_points : bool
            Only return a data frame of individual turbine locations and not
            the rest of the supply-curve table. Defaults to False.
        capacity_col : str
            Column name representing the plant capacity of each supply curve
            point. Defaults to 'capacity_ac_mw'.

        Returns
        -------
        pd.core.frame.DataFrame
            A reV supply curve data frame containing all original farm points
            except one that is replaced with individual turbine entries.
        """
        # Separate target row
        row = self.df.loc[self.index]
        df = self.df.copy()

        # Get coordinates from equal area projection
        new_rows = self.unpack_row(row, capacity_col=capacity_col)

        # Build new data frame
        rdf = pd.DataFrame(new_rows)
        rdf = rdf.reset_index(drop=True)
        rdf.index = df.index[-1] + rdf.index + 1

        # Convert back to WGS84
        rdf = self.to_wgs(rdf)

        if drop_sc_points:
            return rdf

        # Replace the original row with one of the new rows.
        df.iloc[self.index] = rdf.iloc[-1]
        rdf = rdf.iloc[:-1]
        df = pd.concat([df, rdf])

        return df

    def unpack_all(self, capacity_col="capacity_ac_mw", just_coords=False,
                   n_workers=None, n_chunks=500):
        """Unpack all turbines in all sc points."""
        if n_workers is None:
            n_workers = mp.cpu_count()
        df = self.df[self.df[capacity_col] > 0]
        if just_coords:
            df = df[["sc_point_gid", "latitude", "longitude", capacity_col,
                     "turbine_y_coords", "turbine_x_coords"]]
        chunks = np.array_split(df, n_chunks)
        new_entries = []
        with mp.Pool(n_workers) as pool:
            mapper = pool.imap(
                partial(self._unpack_all, capacity_col=capacity_col),
                chunks
            )
            for entry in tqdm(mapper, total=len(chunks)):
                new_entries.append(entry)
        ndf = pd.concat(new_entries)
        ndf = ndf.reset_index(drop=True)
        return ndf

    def _unpack_all(self, df, capacity_col="capacity_ac_mw"):
        """Unpack all turbines in all sc points."""
        new_rows = []
        for _, row in df.iterrows():
            out = self.unpack_row(row, capacity_col=capacity_col)
            new_rows += out
        ndf = pd.DataFrame(new_rows)
        ndf = ndf.reset_index(drop=True)
        return ndf

    def _declick(self, clicksel):
        """Set needed values from click selection as attributes."""
        if self.clicksel:
            point = clicksel["points"][0]
            self.index = point["pointIndex"]
            self.text = point["hovertext"].replace("<br>", "")
        elif self.sc_point_gid:
            query = self.df["sc_point_gid"] == self.sc_point_gid
            self.index = self.df.index[query].values[0]
        self.county = self.df.loc[self.index, "county"]
        self.state = self.df.loc[self.index, "state"]


def batch_unpack_from_supply_curve(sc_df, n_workers=1):
    """Batch functionality to unpack all turbines from a supply curve
        dataframe.

        Parameters
        ----------
        sc_df : pd.core.frame.DataFrame
            A reV supply curve pandas data frame.
        n_workers : int
            Number of workers to use for parallel processing.
            Default is 1 which will run in serial (and will be slow).
            It is recommended to use set n_workers >= 4 for speed.

        Returns
        -------
        geopandas.geodataframe.GeoDataFrame
            A GeoDataFrame containing point locations for all turbines unpacked
            from the source reV supply curve data frame.
    """

    # cap nb_workers to the total CPUs on the machine/node
    n_workers = min(n_workers, cpu_count())

    if n_workers > 1:
        # initialize functionality for parallela dataframe.apply
        pandarallel.initialize(
            progress_bar=True, nb_workers=n_workers, use_memory_fs=False)

    # filter out supply curve points with no capacity (i.e., no turbines)
    sc_developable_df = sc_df[sc_df['capacity'] > 0].copy()
    # reset the index because otherwise the unpacker will get messed up
    sc_developable_df.reset_index(drop=True, inplace=True)

    # unpack the turbine coordinates
    if n_workers > 1:
        # run in parallel
        all_turbines = sc_developable_df.parallel_apply(
            lambda row:
                BespokeUnpacker(
                    sc_developable_df,
                    sc_point_gid=row['sc_point_gid']
                ).unpack_turbines(drop_sc_points=True),
            axis=1
        )
    else:
        # run in serial
        all_turbines = sc_developable_df.apply(
            lambda row:
                BespokeUnpacker(
                    sc_developable_df,
                    sc_point_gid=row['sc_point_gid']
                ).unpack_turbines(drop_sc_points=True),
            axis=1
        )

    # stack the results back into a single df
    all_turbines_df = pd.concat(all_turbines.tolist())

    # extract the geometries
    all_turbines_df['geometry'] = all_turbines_df.apply(
        lambda row: geometry.Point(
            row['longitude'],
            row['latitude']
        ),
        axis=1
    )
    # turn into a geodataframe
    all_turbines_gdf = gpd.GeoDataFrame(all_turbines_df, crs='EPSG:4326')

    return all_turbines_gdf
