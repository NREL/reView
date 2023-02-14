import click
import logging
import pathlib
import pandas as pd
from reView.utils.bespoke import batch_unpack_from_supply_curve

from reView import __version__

logger = logging.getLogger(__name__)

@click.group()
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
              help=('Path to bespoke wind supply curve CSV file created by reV'))
@click.option('--out_gpkg', '-o', required=True,
              prompt='Path to output geopackage.',
              type=click.Path(),
              help=('Path to regions shapefile containing labeled geometries'))
@click.option('--n_workers', '-n', default=1, type=int,
              show_default=True,
              required=False,
              help=('Number of workers to use for parallel processing.'
                    'Default is 1 which will unpack turbines from each supply curve '
                    'grid cell in parallel. This will be slow. It is recommended to use '
                    'at least 4 workers if possible'
              ))
@click.option('--overwrite', default=False,
              show_default=True,
              required=False,
              is_flag=True,
              help=('Overwrite output geopackage if it already exists. Default is False.'))
def unpack_bespoke_turbines_from_supply_curve(supply_curve_csv, out_gpkg, n_workers, overwrite):

    supply_curve_csv_path = pathlib.Path(supply_curve_csv)
    if not supply_curve_csv_path.exists:
        raise FileExistsError(f"Input supply_curve_csv {supply_curve_csv} does not exist.")
    
    supply_curve_df = pd.read_csv(supply_curve_csv_path)

    turbines_gdf = batch_unpack_from_supply_curve(supply_curve_df, n_workers=n_workers)
    
    out_gpkg_path = pathlib.Path(out_gpkg)
    if out_gpkg_path.exists() and overwrite is False:
        raise FileExistsError(f"Output geopackage {out_gpkg} already exists. Use --overwrite to overwrite the existing dataset.")
    
    turbines_gdf.to_file(out_gpkg_path, driver='GPKG')    

