# -*- coding: utf-8 -*-
"""
reView command line interface (CLI).
"""
import json
import logging
from pathlib import Path
import warnings

import click
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import mapclassify as mc

from reView.utils.bespoke import batch_unpack_from_supply_curve
from reView.utils import characterizations, plots
from reView.utils.functions import find_capacity_column
from reView import __version__, REVIEW_DATA_DIR

logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = {
    "max_content_width": 100,
    "terminal_width": 100
}
TECH_CHOICES = ["wind", "solar"]
DEFAULT_BOUNDARIES = Path(REVIEW_DATA_DIR).joinpath(
    "boundaries",
    "ne_50m_admin_1_states_provinces_lakes_conus.geojson"
)
IMAGE_FORMAT_CHOICES = ["png", "pdf", "svg", "jpg"]


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, verbose):
    """reView command line interface."""
    ctx.ensure_object(dict)
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


@main.command()
@click.option('--supply_curve_csv', '-i', required=True,
              type=click.Path(exists=True),
              help='Path to bespoke wind supply curve CSV file created by reV')
@click.option('--out_gpkg', '-o', required=True,
              type=click.Path(),
              help='Path to regions shapefile containing labeled geometries')
@click.option('--n_workers', '-n', default=1, type=int,
              show_default=True,
              required=False,
              help='Number of workers to use for parallel processing.'
                   'Default is 1 which will unpack turbines from each supply '
                   'curve grid cell in parallel. This will be slow. It is '
                   'recommended to use at least 4 workers if possible')
@click.option('--overwrite', default=False,
              show_default=True,
              required=False,
              is_flag=True,
              help='Overwrite output geopackage if it already exists. '
                   'Default is False.')
def unpack_turbines(
        supply_curve_csv, out_gpkg, n_workers, overwrite
):
    """
    Unpack individual turbines from each reV project site in a reV
    supply curve CSV, produced using "bespoke" (i.e., SROM) turbine placement.
    """

    supply_curve_csv_path = Path(supply_curve_csv)
    if not supply_curve_csv_path.exists:
        raise FileExistsError(
            f"Input supply_curve_csv {supply_curve_csv} does not exist."
        )
    supply_curve_df = pd.read_csv(supply_curve_csv_path)

    turbines_gdf = batch_unpack_from_supply_curve(
        supply_curve_df, n_workers=n_workers)

    out_gpkg_path = Path(out_gpkg)
    if out_gpkg_path.exists() and overwrite is False:
        raise FileExistsError(
            f"Output geopackage {out_gpkg} already exists. "
            "Use --overwrite to overwrite the existing dataset.")

    if overwrite is True:
        Path(out_gpkg_path).unlink(missing_ok=True)

    turbines_gdf.to_file(out_gpkg_path, driver='GPKG')


@main.command()
@click.option('--supply_curve_csv', '-i', required=True,
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help='Path to bespoke wind supply curve CSV file created by reV')
@click.option('--char_map', '-m', required=True,
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help='Path to JSON file storing characterization map')
@click.option('--out_csv', '-o', required=True,
              type=click.Path(dir_okay=False),
              help='Path to CSV to store results')
@click.option('--cell_size', '-c', required=False,
              default=90.,
              type=float,
              help=('Cell size in meters of characterization layers. '
                    'Default is 90.'))
@click.option('--overwrite', default=False,
              show_default=True,
              required=False,
              is_flag=True,
              help='Overwrite output CSV if it already exists. '
                   'Default is False.')
def unpack_characterizations(
    supply_curve_csv, char_map, out_csv, cell_size=90., overwrite=False
):
    """
    Unpacks characterization data from the input supply curve dataframe,
    converting values from embedded JSON strings to new standalone columns,
    and saves out a new version of the supply curve with these columns
    included.
    """

    supply_curve_df = pd.read_csv(supply_curve_csv)
    with open(char_map, 'r') as f:
        characterization_map = json.load(f)
    char_df = characterizations.unpack_characterizations(
        supply_curve_df, characterization_map, cell_size
    )

    if overwrite is True:
        Path(out_csv).unlink(missing_ok=True)
    char_df.to_csv(out_csv, header=True, index=False, mode="x")


def validate_breaks_scheme(ctx, param, value):
    # pylint: disable=unused-argument
    """
    Custom validation for --break-scheme/--techs input to make-maps command.
    Checks that the input value is either one of the valid technologies,
    None, or a string specifying a mapclassifier classifier and (optionally)
    its keyword arguments, delimited by a colon (e.g.,
    'equalinterval:{"k":10}'.)

    Parameters
    ----------
    ctx : click.core.Context
        Unused
    param : click.core.Option
        Unused
    value : [str, None]
        Value of the input parameter

    Returns
    -------
    [str, None, tuple]
        Returns one of the following:
         - a string specifying a technology name
         - None type (if the input value was not specified)
         - a tuple of the format (str, dict), where the string is the name
         of a mapclassify classifier and the dictionary are the keyword
         arguments to be passed to that classfier.

    Raises
    ------
    click.BadParameter
        A BadParameter exception will be raised if either of the following
        cases are encountered:
        - an invalid classifier name is specified
        - the kwargs do not appear to be valid JSON
    """

    if value in TECH_CHOICES or value is None:
        return value

    classifier_inputs = value.split(":", maxsplit=1)
    classifier = classifier_inputs[0].lower()
    if classifier not in [c.lower() for c in mc.CLASSIFIERS]:
        raise click.BadParameter(
            f"Classifier {classifier} not recognized as one of the valid "
            f"options: {mc.CLASSIFIERS}."
        )

    classifier_kwargs = {}
    if len(classifier_inputs) == 2:
        try:
            classifier_kwargs = json.loads(classifier_inputs[1])
        except json.decoder.JSONDecodeError as e:
            raise click.BadParameter(
                "Keyword arguments for classifier must be formated as valid "
                "JSON."
            ) from e

    return classifier, classifier_kwargs


@main.command()
@click.option('--supply_curve_csv', '-i', required=True,
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help='Path to supply curve CSV file.')
@click.option("--breaks-scheme",
              "-S",
              required=False,
              type=click.STRING,
              callback=validate_breaks_scheme,
              help=("The format for this option is either 'wind' or 'solar', "
                    "for the hard-coded breaks for those technologies, or "
                    "'<classifier-name>:<classifier-kwargs>' where "
                    "<classifier-name> is one of the valid classifiers from "
                    "the mapclassify package "
                    "(see https://pysal.org/mapclassify/api.html#classifiers) "
                    "and <classifier-kwargs> is an optional set of keyword "
                    "arguments to pass to the classifier function, formatted "
                    "as a JSON. So, a valid input would be "
                    "'equalinterval:{\"k\": 10}' (this would produce 10 equal "
                    "interval breaks). Note that this should all be entered "
                    "as a single string, wrapped in single quotes. "
                    "Alternatively the user can specify just 'equalinterval' "
                    "without the kwargs JSON for the equal interval "
                    "classifier to be used with its default 5 bins (in this "
                    "case, wrapping the string in single quotes is optional) "
                    "The --breaks-scheme option must be specified unless the "
                    "legacy --tech option is used instead."))
@click.option("--tech",
              "-t",
              required=False,
              type=click.STRING,
              callback=validate_breaks_scheme,
              help="Alias for --breaks-scheme. For backwards compatibility "
              "only.")
@click.option('--out_folder', '-o', required=True,
              type=click.Path(exists=False, dir_okay=True, file_okay=False),
              help='Path to output folder for maps.')
@click.option('--boundaries', '-b', required=False,
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              default=DEFAULT_BOUNDARIES,
              # noqa: E126
              help=('Path to vector dataset with the boundaries to map. '
                    'Default is to use state boundaries for CONUS from '
                    'Natural Earth (1:50m scale), which is suitable for CONUS '
                    'supply curves. For other region, it is recommended to '
                    'provide a more appropriate boundaries dataset. The input '
                    'vector dataset can be in CRS.'
                    ))
@click.option('--keep-zero', '-K', default=False,
              required=False,
              is_flag=True,
              help='Keep zero capacity supply curve project sites. These '
                   'sites are dropped by default.')
@click.option('--dpi', '-d', required=False,
              default=600,
              type=click.IntRange(min=0),
              help='Dots-per-inch (DPI) for output images. Default is 600.')
@click.option("--out-format", "-F", required=False,
              default="png",
              type=click.Choice(IMAGE_FORMAT_CHOICES, case_sensitive=True),
              help="Output format for images. Default is ``png`` "
              f"Valid options are: {IMAGE_FORMAT_CHOICES}.")
@click.option('--drop-legend', '-D', default=False,
              required=False,
              is_flag=True,
              help='Drop legend from map. Legend is shown by default.')
def make_maps(
    supply_curve_csv, breaks_scheme, tech, out_folder, boundaries, keep_zero,
    dpi, out_format, drop_legend
):
    """
    Generates standardized, presentation-quality maps for the input supply
    curve, including maps for each of the following attributes:
    Capacity (capacity), All-in LCOE (total_lcoe), Project LCOE (mean_lcoe),
    LCOT (lcot), Capacity Density (derived column) [wind only]
    """

    if tech is None and breaks_scheme is None:
        raise click.MissingParameter(
            "Either --breaks-scheme or --tech must be specified."
        )
    if tech is not None and breaks_scheme is not None:
        warnings.warn(
            "Both --breaks-scheme and --tech were specified: "
            "input for --tech will be ignored"
        )
    if tech is not None and breaks_scheme is None:
        breaks_scheme = tech

    out_path = Path(out_folder)
    out_path.mkdir(exist_ok=True, parents=False)

    supply_curve_df = pd.read_csv(supply_curve_csv)

    cap_col = find_capacity_column(supply_curve_df)

    if keep_zero:
        supply_curve_subset_df = supply_curve_df
    else:
        supply_curve_subset_df = supply_curve_df[
            supply_curve_df[cap_col] > 0
        ].copy()

    supply_curve_gdf = gpd.GeoDataFrame(
        supply_curve_subset_df,
        geometry=gpd.points_from_xy(
            x=supply_curve_subset_df['longitude'],
            y=supply_curve_subset_df['latitude']
        ),
        crs="EPSG:4326"
    )
    supply_curve_gdf["capacity_density"] = (
        supply_curve_gdf[cap_col] /
        supply_curve_gdf["area_sq_km"].replace(0, np.nan)
    ).replace(np.nan, 0)

    boundaries_gdf = gpd.read_file(boundaries)
    boundaries_gdf.to_crs("EPSG:4326", inplace=True)
    boundaries_singlepart_gdf = boundaries_gdf.explode(index_parts=True)

    boundaries_dissolved = boundaries_gdf.unary_union
    background_gdf = gpd.GeoDataFrame(
        {"geometry": [boundaries_dissolved]},
        crs=boundaries_gdf.crs
    ).explode(index_parts=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        map_extent = background_gdf.buffer(0.01).total_bounds

    map_vars = {
        "total_lcoe": {
            "breaks": [25, 30, 35, 40, 45, 50, 60, 70],
            "cmap": 'YlGn',
            "legend_title": "All-in LCOE ($/MWh)"
        },
        "mean_lcoe": {
            "breaks": [25, 30, 35, 40, 45, 50, 60, 70],
            "cmap": 'YlGn',
            "legend_title": "Project LCOE ($/MWh)"
        },
        "lcot": {
            "breaks": [5, 10, 15, 20, 25, 30, 35, 40, 50],
            "cmap": 'YlGn',
            "legend_title": "LCOT ($/MWh)",
        },
        "area_sq_km": {
            "breaks": [5, 10, 25, 50, 100, 120],
            "cmap": "BuPu",
            "legend_title": "Developable Area (sq km)"
        },
        cap_col: {
            "breaks": None,
            "cmap": 'PuRd',
            "legend_title": "Capacity (MW)"
        },
        "capacity_density": {
            "breaks": None,
            "cmap": 'PuRd',
            "legend_title": "Capacity Density (MW/sq km)"
        }
    }

    if breaks_scheme == "solar":
        out_suffix = breaks_scheme
        ac_cap_col = find_capacity_column(
            supply_curve_df,
            cap_col_candidates=["capacity_ac", "capacity_mw_ac"]
        )
        map_vars.update({
            cap_col: {
                "breaks": [100, 500, 1000, 2000, 3000, 4000],
                "cmap": 'YlOrRd',
                "legend_title": "Capacity DC (MW)"
            },
            ac_cap_col: {
                "breaks": [100, 500, 1000, 2000, 3000, 4000],
                "cmap": 'YlOrRd',
                "legend_title": "Capacity AC (MW)"
            },
            "capacity_density": {
                "breaks": [30, 40, 50, 60, 70],
                "cmap": 'YlOrRd',
                "legend_title": "Capacity Density (MW/sq km)"
            }
        })
    elif breaks_scheme == "wind":
        out_suffix = breaks_scheme
        map_vars.update({
            cap_col: {
                "breaks": [60, 120, 180, 240, 275],
                "cmap": 'Blues',
                "legend_title": "Capacity (MW)"
            },
            "capacity_density": {
                "breaks": [2, 3, 4, 5, 6, 10],
                "cmap": 'Blues',
                "legend_title": "Capacity Density (MW/sq km)"
            }
        })
    else:
        classifier, classifier_kwargs = breaks_scheme
        out_suffix = classifier
        # pylint: disable=consider-using-dict-items,
        # consider-iterating-dictionary
        for map_var in map_vars:
            scheme = mc.classify(
                supply_curve_gdf[map_var], classifier, **classifier_kwargs
            )
            breaks = scheme.bins
            map_vars[map_var]["breaks"] = breaks.tolist()[0:-1]

    for map_var, map_settings in tqdm.tqdm(map_vars.items()):
        g = plots.map_geodataframe_column(
            supply_curve_gdf,
            map_var,
            color_map=map_settings.get("cmap"),
            breaks=map_settings.get("breaks"),
            map_title=None,
            legend_title=map_settings.get("legend_title"),
            background_df=background_gdf,
            boundaries_df=boundaries_singlepart_gdf,
            extent=map_extent,
            layer_kwargs={"s": 2.0, "linewidth": 0, "marker": "o"},
            legend_kwargs={
                "marker": "s",
                "frameon": False,
                "bbox_to_anchor": (1, 0.5),
                "loc": "center left"
            },
            legend=(not drop_legend)
        )
        bbox = g.get_tightbbox(g.figure.canvas.get_renderer())
        fig_height = g.figure.get_figheight()
        g.figure.set_figwidth(fig_height * bbox.width / bbox.height)
        plt.tight_layout(pad=0.1)

        out_image_name = f"{map_var}_{out_suffix}.{out_format}"
        out_image_path = out_path.joinpath(out_image_name)
        g.figure.savefig(out_image_path, dpi=dpi, transparent=True)
        plt.close(g.figure)


@main.command()
@click.option('--supply_curve_csv', '-i', required=True,
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help='Path to supply curve CSV file.')
@click.option('--out_folder', '-o', required=True,
              type=click.Path(exists=False, dir_okay=True, file_okay=False),
              help='Path to output folder for maps.')
@click.option('--column', '-c', required=True,
              type=str,
              help='Column to map')
@click.option('--colormap', '-C', required=False,
              type=str,
              default=None,
              help=('Color map to use for the column. Refer to https://'
                    'matplotlib.org/stable/tutorials/colors/colormaps.html'
                    ' for valid options. If not specified, the viridis '
                    'colormap will be applied.'))
@click.option('--legend_title', '-T', required=False,
              type=str,
              default=None,
              help=('Title to use for the map legend. '
                    'If not provided, legend title will be the column name'))
@click.option('--legend_breaks', '-B', required=False,
              type=str,
              default=None,
              help=('Breaks to use for the map legend. Should be formatted '
                    'like a list in quotes, e.g. : "[10, 50, 100, 150]". If '
                    'not provided, a 5-class quantile classification will be '
                    'used to derive the breaks.'))
@click.option('--boundaries', '-b', required=False,
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              default=DEFAULT_BOUNDARIES,
              # noqa: E126
              help=('Path to vector dataset with the boundaries to map. '
                    'Default is to use state boundaries for CONUS from '
                    'Natural Earth (1:50m scale), which is suitable for CONUS '
                    'supply curves. For other region, it is recommended to '
                    'provide a more appropriate boundaries dataset. The input '
                    'vector dataset can be in CRS.'
                    ))
@click.option('--keep_zero', '-K', default=False,
              required=False,
              is_flag=True,
              help='Keep zero capacity supply curve project sites. These '
                   'Sites are dropped by default.')
@click.option('--dpi', '-d', required=False,
              default=600,
              type=click.IntRange(min=0),
              help='Dots-per-inch (DPI) for output images. Default is 600.')
@click.option("--out-format", "-F", required=False,
              default="png",
              type=click.Choice(IMAGE_FORMAT_CHOICES, case_sensitive=True),
              help="Output format for images. Default is ``png`` "
                   f"Valid options are: {IMAGE_FORMAT_CHOICES}.")
@click.option('--drop-legend', '-D', default=False,
              required=False,
              is_flag=True,
              help='Drop legend from map. Legend is shown by default.')
@click.option('--boundaries_kwargs', '-bk', required=False,
              type=str,
              default=None,
              help=('Boundaries keyword arguments to change styling of '
                    'boundary lines. For example, to make boundaries 2x '
                    'thicker and black instead of white, specify: '
                    '\'{"linewidth": 1.0, "zorder": 1, "edgecolor": "black"}\''
                    ))
def map_column(
    supply_curve_csv, out_folder, column, colormap, legend_title,
    legend_breaks, boundaries, keep_zero, dpi, out_format, drop_legend,
    boundaries_kwargs
):
    # pylint: disable=too-many-arguments
    """
    Generates a single map from an input supply curve for the specified column,
    with basic options for formatting.
    """

    out_path = Path(out_folder)
    out_path.mkdir(exist_ok=True, parents=False)

    supply_curve_df = pd.read_csv(supply_curve_csv)
    if column not in supply_curve_df.columns:
        raise KeyError(
            f"Column {column} could not be found in input supply curve."
        )

    if keep_zero:
        supply_curve_subset_df = supply_curve_df
    else:
        cap_col = find_capacity_column(supply_curve_df)
        supply_curve_subset_df = supply_curve_df[
            supply_curve_df[cap_col] > 0
        ].copy()

    supply_curve_gdf = gpd.GeoDataFrame(
        supply_curve_subset_df,
        geometry=gpd.points_from_xy(
            x=supply_curve_subset_df['longitude'],
            y=supply_curve_subset_df['latitude']
        ),
        crs="EPSG:4326"
    )

    boundaries_gdf = gpd.read_file(boundaries)
    boundaries_gdf.to_crs("EPSG:4326", inplace=True)
    boundaries_singlepart_gdf = boundaries_gdf.explode(index_parts=True)

    boundaries_dissolved = boundaries_gdf.unary_union
    background_gdf = gpd.GeoDataFrame(
        {"geometry": [boundaries_dissolved]},
        crs=boundaries_gdf.crs
    ).explode(index_parts=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        map_extent = background_gdf.buffer(0.01).total_bounds

    if legend_breaks is None:
        breaks = None
    else:
        try:
            if not legend_breaks.startswith('['):
                raise ValueError("Invalid input: does not start with '['.")
            if not legend_breaks.endswith("]"):
                raise ValueError("Invalid input: does not start with ']'.")
            breaks = [
                float(b.strip()) for b in legend_breaks[1:-1].split(',')
            ]
        except Exception as e:
            raise ValueError(
                "Input legend_breaks could not be parsed as a list of floats. "
                f"The following error was encountered: {e}"
            ) from e

    if legend_title is None:
        legend_title = column

    if boundaries_kwargs is not None:
        boundaries_kwargs_dict = json.loads(boundaries_kwargs)
    else:
        boundaries_kwargs_dict = None

    g = plots.map_geodataframe_column(
        supply_curve_gdf,
        column,
        color_map=colormap,
        breaks=breaks,
        map_title=None,
        legend_title=legend_title,
        background_df=background_gdf,
        boundaries_df=boundaries_singlepart_gdf,
        extent=map_extent,
        layer_kwargs={"s": 2.0, "linewidth": 0, "marker": "o"},
        legend_kwargs={
            "marker": "s",
            "frameon": False,
            "bbox_to_anchor": (1, 0.5),
            "loc": "center left"
        },
        boundaries_kwargs=boundaries_kwargs_dict,
        legend=(not drop_legend)
    )
    bbox = g.get_tightbbox(g.figure.canvas.get_renderer())
    fig_height = g.figure.get_figheight()
    g.figure.set_figwidth(fig_height * bbox.width / bbox.height)
    plt.tight_layout(pad=0.1)

    out_image_name = f"{column}.{out_format}"
    out_image_path = out_path.joinpath(out_image_name)
    g.figure.savefig(out_image_path, dpi=dpi, transparent=True)
    plt.close(g.figure)


@main.command()
@click.argument('supply_curve_csv',
                type=click.Path(exists=True, dir_okay=False, file_okay=True)
                )
@click.option('--column', '-c', required=True, multiple=True,
              default=None,
              type=click.STRING,
              help=(
                  "Value column from the input CSV to plot. Multiple value "
                  "columnscan be specified: e.g., -c area_sq_km -c capacity_mw"
                  ))
@click.option('--nbins', '-N', required=False,
              default=20,
              type=click.IntRange(min=1),
              help=("Number of bins to use in the histogram. If not "
                    "specified, default is 20 bins."))
@click.option('--width', '-W', required=False,
              default=None,
              type=click.IntRange(min=0, max=500),
              help=("Width of output histogram. If not specified, default "
                    "width is 80% of the terminal width."))
@click.option('--height', '-H', required=False,
              default=None,
              type=click.IntRange(min=0, max=500),
              help=("Height of output histogram. If not specified, default "
                    "height is the smaller of 20% of the terminal width or "
                    "100% of the terminal height."))
def histogram(supply_curve_csv, column, nbins, width, height):
    """
    Plots a histogram in the terminal for the specified column(s) from the
    input SUPPLY_CURVE_CSV.
    """

    df = pd.read_csv(supply_curve_csv)
    for column_name in column:
        try:
            plots.ascii_histogram(
                df, column_name, nbins=nbins, width=width, height=height
            )
            print("\n")
        except TypeError as e:
            print(f"Unable to plot column '{column_name}': {e}")
