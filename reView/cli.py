# -*- coding: utf-8 -*-
"""
reView command line interface (CLI).
"""
import json
import logging
import pathlib
import click
import pandas as pd
from reView.utils.bespoke import batch_unpack_from_supply_curve
from reView.utils import characterizations
from reView import __version__

logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = {
    "max_content_width": 9999,
    "terminal_width": 9999
}


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
              prompt='Path to input bespoke wind supply curve CSV file',
              type=click.Path(exists=True),
              help='Path to bespoke wind supply curve CSV file created by reV')
@click.option('--out_gpkg', '-o', required=True,
              prompt='Path to output geopackage.',
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

    supply_curve_csv_path = pathlib.Path(supply_curve_csv)
    if not supply_curve_csv_path.exists:
        raise FileExistsError(
            f"Input supply_curve_csv {supply_curve_csv} does not exist."
        )
    supply_curve_df = pd.read_csv(supply_curve_csv_path)

    turbines_gdf = batch_unpack_from_supply_curve(
        supply_curve_df, n_workers=n_workers)

    out_gpkg_path = pathlib.Path(out_gpkg)
    if out_gpkg_path.exists() and overwrite is False:
        raise FileExistsError(
            f"Output geopackage {out_gpkg} already exists. "
            "Use --overwrite to overwrite the existing dataset.")

    if overwrite is True:
        out_gpkg_path.unlink(missing_ok=True)

    turbines_gdf.to_file(out_gpkg_path, driver='GPKG')


@main.command()
@click.option('--supply_curve_csv', '-i', required=True,
              prompt='Path to input bespoke wind supply curve CSV file',
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help='Path to bespoke wind supply curve CSV file created by reV')
@click.option('--char_map', '-m', required=True,
              prompt='Path to JSON file storing characterization map',
              type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help='Path to JSON file storing characterization map')
@click.option('--out_csv', '-o', required=True,
              prompt='Path to output csv.',
              type=click.Path(dir_okay=False),
              help='Path to CSV to store results')
@click.option('--cell_size', '-c', required=False,
              default=90.,
              type=float,
              help='Cell size in meters of characterization layers')
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
    char_df.to_csv(out_csv, header=True, index=False)
