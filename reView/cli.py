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
import matplotlib.pyplot as plt
import tqdm

from reView.utils.bespoke import batch_unpack_from_supply_curve
from reView.utils import characterizations, plots
from reView import __version__, REVIEW_DATA_DIR

logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = {
    "max_content_width": 9999,
    "terminal_width": 9999
}
TECH_CHOICES = ["wind", "solar"]
DEFAULT_BOUNDARIES = Path(REVIEW_DATA_DIR).joinpath(
    "boundaries",
    "ne_50m_admin_1_states_provinces_lakes_conus.geojson"
)


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
        out_gpkg_path.unlink(missing_ok=True)

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
        out_csv.unlink(missing_ok=True)
    char_df.to_csv(out_csv, header=True, index=False, mode="x")


@main.command()
@click.option('--supply_curve_csv', '-i', required=True,
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help='Path to supply curve CSV file.')
@click.option("--tech",
              "-t",
              required=True,
              type=click.Choice(TECH_CHOICES, case_sensitive=False),
              help="Technology choice for ordinances to export. "
              f"Valid options are: {TECH_CHOICES}.")
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
                    'provide a more appropriate boundaries dataset.'
                    ))
@click.option('--dpi', '-d', required=False,
              default=600,
              type=click.IntRange(min=0),
              help='Dots-per-inch (DPI) for output images. Default is 600.')
def make_maps(
    supply_curve_csv, tech, out_folder, boundaries, dpi
):
    """
    Generates standardized, presentation-quality maps for the input supply
    curve, including maps for each of the following attributes:
    Capacity (capacity), All-in LCOE (total_lcoe), Project LCOE (mean_lcoe),
    LCOT (lcot), Capacity Density (derived column) [wind only]
    """

    out_path = Path(out_folder)
    out_path.mkdir(exist_ok=True, parents=False)

    supply_curve_df = pd.read_csv(supply_curve_csv)
    supply_curve_gdf = gpd.GeoDataFrame(
        supply_curve_df,
        geometry=gpd.points_from_xy(
            x=supply_curve_df['longitude'], y=supply_curve_df['latitude']
        ),
        crs="EPSG:4326"
    )

    boundaries_gdf = gpd.read_file(boundaries)
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
        }
    }
    if tech == "solar":
        map_vars.update({
            "capacity": {
                "breaks": [100, 500, 1000, 2000, 3000, 4000],
                "cmap": 'YlOrRd',
                "legend_title": "Capacity (MW)"
            }
        })
    elif tech == "wind":
        supply_curve_gdf["capacity_density"] = (
            supply_curve_gdf["capacity"] / supply_curve_gdf["area_sq_km"]
        )
        map_vars.update({
            "capacity": {
                "breaks": [100, 125, 150, 175, 200, 225],
                "cmap": 'Blues',
                "legend_title": "Capacity (MW)"
            },
            "capacity_density": {
                "breaks": [2, 3, 4, 5, 6, 10],
                "cmap": 'Blues',
                "legend_title": "Capacity Density (MW/sq km)"
            }
        })
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
            layer_kwargs={"s": 1.25, "linewidth": 0, "marker": "o"},
            legend_kwargs={
                "marker": "s",
                "frameon": False,
                "bbox_to_anchor": (1, 0.5),
                "loc": "center left"
            }
        )
        plt.tight_layout()

        out_png_name = f"{map_var}_{tech}.png"
        out_png = out_path.joinpath(out_png_name)
        g.figure.savefig(out_png, dpi=dpi)
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
                    'like a list, e.g. : "[10, 50, 100, 150]". If not '
                    'provided, a 5-class quantile classification will be used '
                    'to derive the breaks.'))
@click.option('--boundaries', '-b', required=False,
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              default=DEFAULT_BOUNDARIES,
              # noqa: E126
              help=('Path to vector dataset with the boundaries to map. '
                    'Default is to use state boundaries for CONUS from '
                    'Natural Earth (1:50m scale), which is suitable for CONUS '
                    'supply curves. For other region, it is recommended to '
                    'provide a more appropriate boundaries dataset.'
                    ))
@click.option('--dpi', '-d', required=False,
              default=600,
              type=click.IntRange(min=0),
              help='Dots-per-inch (DPI) for output images. Default is 600.')
def map_column(
    supply_curve_csv, out_folder, column, colormap=None, legend_title=None,
    legend_breaks=None, boundaries=DEFAULT_BOUNDARIES, dpi=600
):
    # pylint: disable=raise-missing-from
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
    supply_curve_gdf = gpd.GeoDataFrame(
        supply_curve_df,
        geometry=gpd.points_from_xy(
            x=supply_curve_df['longitude'], y=supply_curve_df['latitude']
        ),
        crs="EPSG:4326"
    )

    boundaries_gdf = gpd.read_file(boundaries)
    boundaries_singlepart_gdf = boundaries_gdf.explode(index_parts=True)

    boundaries_dissolved = boundaries_gdf.unary_union
    background_gdf = gpd.GeoDataFrame(
        {"geometry": [boundaries_dissolved]},
        crs=boundaries_gdf.crs
    ).explode(index_parts=False)

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
            )

    if legend_title is None:
        legend_title = column

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
        layer_kwargs={"s": 1.25, "linewidth": 0, "marker": "o"},
        legend_kwargs={
            "marker": "s",
            "frameon": False,
            "bbox_to_anchor": (1, 0.5),
            "loc": "center left"
        }
    )
    plt.tight_layout()

    out_png_name = f"{column}.png"
    out_png = out_path.joinpath(out_png_name)
    g.figure.savefig(out_png, dpi=dpi)
    plt.close(g.figure)
