# -*- coding: utf-8 -*-
"""Methods for creating static, report quality graphics.

@author: Mike Gleason
"""
import warnings

import numpy as np
import mapclassify as mc
from matplotlib.patheffects import SimpleLineShadow, Normal
import geoplot as gplt
import plotext
from pandas.api.types import is_numeric_dtype


# pylint: disable=no-member, arguments-differ
class YBFixedBounds(np.ndarray):
    """
    Helper class for use with a mapclassify classifer. This can used to
    overwrite the ``yb`` property of a classifier so that the .max() and .min()
    methods return preset values rather than the maximum and minimum break
    labels corresponding to the range of data.

    This is used in map_supply_curve_column() to ensure that breaks and colors
    shown in the legend are always consistent with the input ``breaks`` rather
    than subject to change based on range of the input ``column``.
    """
    def __new__(cls, input_array, preset_max, preset_min=0):
        """
        Return a new instance of the class with fixed maximum and minimum
        values.

        Parameters
        ----------
        input_array : numpy.ndarray
            Input numpy array, typically sourced from the ``yb`` property of a
            mapclassify classifier.
        preset_max : int
            Maximum value to return when .max() is called. Typically this
            should be set to the classifier ``k`` property, which is the number
            of classes in the classifier.
        preset_min : int, optional
            Minimum value to return when .min() is called. Under most
            circumstances, the default value (0) should be used.

        Returns
        -------
        YBFixedBounds
            New instance of YBFixedBounds with present min() and max() values.
        """
        array = np.asarray(input_array).view(cls)
        array.__dict__.update(
            {
                "_preset_max": preset_max,
                "_preset_min": preset_min,
            }
        )
        return array

    def max(self):
        """Return preset maximum value."""
        return self._preset_max

    def min(self):
        """Return preset minimum value."""
        return self._preset_min


def map_geodataframe_column(  # noqa: C901
    data_df,
    column,
    color_map="viridis",
    breaks=None,
    map_title=None,
    legend_title=None,
    background_df=None,
    boundaries_df=None,
    extent=None,
    boundaries_kwargs=None,
    layer_kwargs=None,
    legend_kwargs=None,
    projection=gplt.crs.AlbersEqualArea(),
    legend=True
):
    # pylint: disable=too-many-arguments,too-many-branches
    """
    Create a cartographic quality map symbolizing the values from an input
    geodataframe, optionally including a background layer (e.g., CONUS
    landmass), a boundary layer (e.g., state boundaries), and various map style
    elements.

    Parameters
    ----------
    data_df : geopandas.geodataframe.GeoDataFrame
        Input GeoDataFrame with values in ``column`` to map. Input geometry
        type must be one of: ``Point``, ``Polygon``, or ``MultiPolygon``.

        If ``background_df`` and ``extent`` are both None, the extent of this
        dataframe will set the overall map extent.
    column : str
        Name of the column in ``data_df`` to plot.
    color_map : [str, matplotlib.colors.Colormap], optional
        Colors to use for mapping the values of ``column``. This can either be
        the name of a colormap or an actual colormap instance.
        By default, the color_map will be "viridis".
    breaks : list, optional
        List of value breaks to use for classifying the values of ``column``
        into the colors of ``color_map``. Break values should be provided in
        ascending order. Values of ``column`` that are below the first break
        or above the last break will be shown in the first and last classes,
        respectively. If not specified, the map will be created using
        a Quantile classification scheme with 5 classes.
    map_title : str, optional
        Title to use for the map, by default None.
    legend_title : str, optional
        Title to use for the legend, by default None.
    background_df : geopandas.geodataframe.GeoDataFrame, optional
        Geodataframe to plot as background, behind ``data_df``. Expected to
        have geometry type of ``Polygon`` or ``MultiPolygon``. A common case
        would be to provide polygons representing the landmass, country, or
        region that you are mapping as the background.

        Providing this layer has the side-effect of creating a dropshadow for
        the whole map, so is generally recommended for nicer styling of the
        output map. Configuration of the display of this layer is not currently
        available to the user.

        If set to ``None`` (default), no background layer will be plotted.

        If specified and ``extent`` is ``None``, the extent of this dataframe
        will set the overall map extent.
    boundaries_df : geopandas.geodataframe.GeoDataFrame, optional
        Geodataframe to plot on the map ``data_df`` as boundaries. Expected
        to have geometry type of ``Polygon`` or ``MultiPolygon``. A common
        case would be to provide polygons for states or other sub-regions of
        interest.

        If set to ``None`` (default), no background layer will be plotted.
    extent : [list, np.ndarray], optional
        Extent to zoom to for displaying the map. Should be of the format:
        [xmin, ymin, xmax, ymax] in the CRS units of data_df. By defaut, this
        is None, which will result in the extent of the map being set based on
        background_df (if provided) or data_df.
    boundaries_kwargs : [dict, None], optional
        Keyword arguments that can be used to configure display of the
        boundaries layer. If not specified (=None), it will default to use
        (``{"linewidth": 0.75, "zorder": 1, "edgecolor": "white"}``), which
        will result in thin white boundaries being plotted underneath the data
        layer. To place these on top, change ``zorder`` to ``2``. For other
        options, refer to
        https://residentmario.github.io/geoplot/user_guide/
        Customizing_Plots.html and https://matplotlib.org/stable/api/_as_gen/
        matplotlib.patches.Polygon.html#matplotlib.patches.Polygon.
    layer_kwargs : [dict, None], optional
        Optional styling to be applied to the data layer. By default None,
        which results in the layer being plotted using the input breaks and
        colormap and no other changes. As an example, you could change the edge
        color and line width of a polygon data layer using by specifying
        ``layer_kwargs={"edgecolor": "gray", "linewidth": 0.5}``. Refer to
        https://residentmario.github.io/geoplot/user_guide/
        Customizing_Plots.html#Cosmetic-parameters for other options.
    legend_kwargs: [dict, None], optional
        Keyword arguments that can be used to configure display of the
        legend. If not specified (=None), it will default to use
        (``legend_kwargs={"marker": "s", "frameon": False,
        "bbox_to_anchor": (1, 0.5), "loc": "center left"}``).
        For more information on the options available, refer to
        https://residentmario.github.io/geoplot/user_guide/
        Customizing_Plots.html#Legend.
    projection: gplt.crs.Base, optional
        Projection to use for creating the map. Default is
        gplt.crs.AlbersEqualArea(). For names of other options, refer to
        https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html.

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxes
        Plot object of the map.

    Raises
    ------
    NotImplementedError
        A NotImplementedError will be raised if ``data_df`` does not have
        a geometry type of ``Point``, ``Polygon``, or ``MultiPolygon``.
    """

    if breaks is None:
        scheme = mc.Quantiles(data_df[column], k=5)
    else:
        # ensure ascending order
        breaks.sort()
        # add inf as the last break to ensure consistent breaks between maps
        if breaks[-1] != np.inf:
            breaks.append(np.inf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scheme = mc.UserDefined(data_df[column], bins=breaks)
        scheme.yb = YBFixedBounds(
            scheme.yb, preset_max=scheme.k - 1, preset_min=0
        )

    if background_df is not None:
        drop_shadow_effects = [
            SimpleLineShadow(
                shadow_color="black", linewidth=0.5, alpha=0.65,
                offset=(1, -1)
            ),
            SimpleLineShadow(
                shadow_color="gray", linewidth=0.5, alpha=0.65,
                offset=(1.5, -1.5)
            ),
            Normal(),
        ]
        if extent is None:
            extent = background_df.total_bounds

        ax = gplt.polyplot(
            background_df,
            facecolor="#bdbdbd",
            linewidth=0,
            edgecolor="#bdbdbd",
            projection=projection,
            extent=extent,
            path_effects=drop_shadow_effects,
        )
    else:
        if extent is None:
            extent = data_df.total_bounds
        ax = None

    input_geom_types = list(set(data_df.geom_type))

    if boundaries_kwargs is None:
        boundaries_kwargs = {
            "linewidth": 0.75,
            "zorder": 1,
            "edgecolor": "white"
        }

    if legend:
        if legend_kwargs is None:
            legend_kwargs = {
                "marker": "s",
                "frameon": False,
                "bbox_to_anchor": (1, 0.5),
                "loc": "center left"
            }
        legend_kwargs["title"] = legend_title
    else:
        legend_kwargs = None

    if input_geom_types == ["Point"]:
        if layer_kwargs is None:
            layer_kwargs = {
                "s": 1.25,  # point size
                "linewidth": 0,
                "marker": "o"
            }
        ax = gplt.pointplot(
            data_df,
            hue=column,
            legend=legend,
            scheme=scheme,
            projection=projection,
            extent=extent,
            ax=ax,
            cmap=color_map,
            legend_kwargs=legend_kwargs,
            **layer_kwargs,
        )
    elif input_geom_types in (["Polygon"], ["MultiPolygon"]):
        ax = gplt.choropleth(
            data_df,
            hue=column,
            legend=legend,
            scheme=scheme,
            projection=gplt.crs.AlbersEqualArea(),
            extent=extent,
            ax=ax,
            cmap=color_map,
            legend_kwargs=legend_kwargs,
            **layer_kwargs,
        )
    else:
        raise NotImplementedError(
            f"Mapping has not been implemented for input with "
            f"geometry types: {input_geom_types}"
        )

    if boundaries_df is not None:
        gplt.polyplot(
            boundaries_df,
            facecolor="None",
            projection=projection,
            extent=extent,
            ax=ax,
            **boundaries_kwargs,
        )

    # fix last legend entry
    if legend is True:
        last_legend_label = ax.legend_.texts[-1]
        new_label = f"> {last_legend_label.get_text().split(' - ')[0]}"
        last_legend_label.set_text(new_label)

        if legend_title is not None:
            ax.legend_.set_title(legend_title)

    if map_title is not None:
        ax.set_title(map_title)

    return ax


def ascii_histogram(df, column, nbins=20, width=None, height=None):
    """
    Prints a histogram for the selected column of the input dataframe to the
    terminal.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    column : str
        Name of column to plot. Must have a numeric dtype.
    nbins : int, optional
        Number of bins to plot in the histogram, by default 20.
    width : int, optional
        Width of the histogram plot (in characters). By default None, which
        will set the width to 80% of the terminal width.
    height : int, optional
        Height of the histogram plot (in characters). By default None, which
        will set the width to the smaller of either 20% of the terminal width
        or 100% of the terminal height.

    Raises
    ------
    TypeError
        _description_
    """

    if not is_numeric_dtype(df[column]):
        raise TypeError("Input column must have a numeric dtype.")

    x_values = df[column].tolist()

    plotext.clear_data()
    plotext.clear_figure()

    if width is None:
        width = plotext.terminal_width() * 0.8
    if height is None:
        height = min(plotext.terminal_width()/5, plotext.terminal_height())

    plotext.hist(x_values, bins=nbins, width=1)
    plotext.plotsize(width, height)
    plotext.axes_color("black")
    plotext.canvas_color("black")
    plotext.ticks_color("gray")
    plotext.xfrequency(nbins+1)
    plotext.title(column)

    plotext.show()
    plotext.clear_data()
    plotext.clear_figure()
