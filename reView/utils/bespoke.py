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

import numpy as np
import pandas as pd
import pyproj

from reView.utils.classes import CountyCode

pyproj.network.set_network_enabled(False)


SPLIT_COLS = ["capacity", "annual_energy-means"]

#pylint: disable=invalid-name,unpacking-non-sequence
class BespokeUnpacker:
    """Methods for manipulating Bespoke reV outputs."""

    def __init__(self, df, clicksel):
        """Initialize BespokeUnpacker object.

        Parameters
        ----------
        df : pd.core.frame.DataFrame
            A reV supply curve pandas data frame.
        clicksel : dict
            Dictionary containing plotly point attributes from a
            scattermapbox point selection.
        """
        self.df = df
        self.clicksel = clicksel
        self.src_crs = "epsg:4326"
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
    def trgt_crs(self):
        """Find an appropriate coordinate reference system for location."""
        code = CountyCode.epsg(self.county, self.state)
        return f"epsg:{code}"

    @property
    def spacing(self):
        """Infer the spacing between points."""
        # Assuming a 128 agg factor for now
        spacing = 11_520
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

    def unpack_turbines(self):
        """Unpack bespoke turbines if possible.

        Returns
        -------
        pd.core.frame.DataFrame
            A reV supply curve data frame containing all original farm points
            except one that is replaced with individual turbine entries.
        """
        # Separate target row
        df = self.df.iloc[self.df.index != self.index]
        row = self.df.iloc[self.index]

        # Get coordinates from equal area projection
        x, y = self.get_xy(row)
        del row["longitude"]
        del row["latitude"]

        # Get bottom left coordinates
        blx = x - (self.spacing / 2)
        bly = y - (self.spacing / 2)

        # Get layout
        xs = json.loads(row["turbine_x_coords"])
        ys = json.loads(row["turbine_y_coords"])
        xs = [x + blx for x in xs]
        ys = [y + bly for y in ys]

        # Build new data frame
        nrows = []
        for i, x in enumerate(xs):
            nrow = row.copy()
            nrow["x"] = x
            nrow["y"] = ys[i]
            nrows.append(nrow)

        rdf = pd.DataFrame(nrows)
        nindex = np.arange(
            self.df.index[-1],
            self.df.index[-1] + len(rdf.index)
        )
        rdf.index = nindex

        # Convert back to WGS84
        rdf = self.to_wgs(rdf)

        # Append to full data frame
        df = pd.concat([self.df, rdf])
        rdf["index"] = rdf.index

        return df

    def _declick(self, clicksel):
        """Set needed values from click selection as attributes."""
        point = clicksel["points"][0]
        self.index = point["pointIndex"]
        self.county = self.df.loc[self.index, "county"]
        self.state = self.df.loc[self.index, "state"]
        self.text = point["hovertext"].replace("<br>", "")
