# Usage Instructions

## Command-Line Interface
The available command-line tools and shared options are available by running: `reView-tools --help`, which will return:

```commandline
Usage: reView-tools [OPTIONS] COMMAND [ARGS]...

  reView command line interface.

Options:
  --version      Show the version and exit.
  -v, --verbose  Flag to turn on debug logging. Default is not verbose.
  --help         Show this message and exit.

Commands:
  histogram                 Plots a histogram in the terminal for the specified column(s) from the input SUPPLY_CURVE_CSV.
  make-maps                 Generates standardized, presentation-quality maps for the input supply curve, including maps for each of the following attributes: Capacity (capacity), All-in LCOE (total_lcoe), Project LCOE (mean_lcoe), LCOT (lcot), Capacity Density (derived column) [wind only]

  map-column                Generates a single map from an input supply curve for the specified column, with basic options for formatting.

  unpack-characterizations  Unpacks characterization data from the input supply curve dataframe, converting values from embedded JSON strings to new standalone columns, and saves out a new version of the supply curve with these columns included.

  unpack-turbines           Unpack individual turbines from each reV project site in a reV supply curve CSV, produced using "bespoke" (i.e., SROM) turbine placement.
```

## Plotting Histograms (in the terminal)
The `histogram` function is intended to make it easy to review the distribution of data contained in a supply curve CSV. It can be used to plot a histogram of the selected CSV column(s) directly in the terminal.

This command can be run according to the following usage:
```commandline
Usage: reView-tools histogram [OPTIONS] SUPPLY_CURVE_CSV

  Plots a histogram in the terminal for the specified column(s) from the input SUPPLY_CURVE_CSV.

Options:
  -c, --column TEXT           Value column from the input CSV to plot. Multiple value columnscan be specified: e.g., -c area_sq_km -c capacity_mw  [required]
  -N, --nbins INTEGER RANGE   Number of bins to use in the histogram. If not specified, default is 20 bins.  [x>=1]
  -W, --width INTEGER RANGE   Width of output histogram. If not specified, default width is 80% of the termimal width.  [0<=x<=500]
  -H, --height INTEGER RANGE  Height of output histogram. If not specified, default height is the smaller of 50% of the terminal width or 100% of the terminal height.  [0<=x<=500]
  --help                      Show this message and exit.
```

For most use cases, users need to only specify the `SUPPLY_CURVE_CSV` and the column or columns they want to plot. For example, to plot the `area_sq_km` and `capacity_mw` columns:
```commandline
reView-tools histogram some_supply_curve.csv -c area_sq_km -c capacity_mw
```

For more advanced cases, the user may want to adjust the number of bins in the histogram using the `--nbins` argument and/or the size of the output plot with the `--width` and/or `--height` arguments.

## Making Standardized Maps
The `make-maps` command can be used to generate a small set of standardized, report/presentation-qualiity maps from an input supply curve. For a solar supply curve, 4 maps are created, including one for each of the following supply curve columns: Capacity (capacity), All-in LCOE (total_lcoe), Project LCOE (mean_lcoe), LCOT (lcot). For a wind supply curve, the same 4 maps are created and, in addition, a map is also created for Capacity Density (a derived column).

This command can be run according to the following usage:
```commandline
Usage: reView-tools make-maps [OPTIONS]

  Generates standardized, presentation-quality maps for the input supply curve, including maps for
  each of the following attributes: Capacity (capacity), All-in LCOE (total_lcoe), Project LCOE
  (mean_lcoe), LCOT (lcot), Capacity Density (derived column) [wind only]

Options:
  -i, --supply_curve_csv FILE     Path to supply curve CSV file.  [required]
  -S, --breaks-scheme TEXT        The format for this option is either 'wind' or 'solar', for the
                                  hard-coded breaks for those technologies, or '<classifier-
                                  name>:<classifier-kwargs>' where <classifier-name> is one of the
                                  valid classifiers from the mapclassify package (see
                                  https://pysal.org/mapclassify/api.html#classifiers) and
                                  <classifier-kwargs> is an optional set of keyword arguments to
                                  pass to the classifier function, formatted as a JSON. So, a valid
                                  input would be 'equalinterval:{"k": 10}' (this would produce 10
                                  equal interval breaks). Note that this should all be entered as a
                                  single string, wrapped in single quotes. Alternatively the user
                                  can specify just 'equalinterval' without the kwargs JSON for the
                                  equal interval classifier to be used with its default 5 bins (in
                                  this case, wrapping the string in single quotes is optional) The
                                  --breaks-scheme option must be specified unless the legacy --tech
                                  option is used instead.
  -t, --tech TEXT                 Alias for --breaks-scheme. For backwards compatibility only.
  -o, --out_folder DIRECTORY      Path to output folder for maps.  [required]
  -b, --boundaries FILE           Path to vector dataset with the boundaries to map. Default is to
                                  use state boundaries for CONUS from Natural Earth (1:50m scale),
                                  which is suitable for CONUS supply curves. For other region, it is
                                  recommended to provide a more appropriate boundaries dataset. The
                                  input vector dataset can be in CRS.
  -K, --keep-zero                 Keep zero capacity supply curve project sites. These sites are
                                  dropped by default.
  -d, --dpi INTEGER RANGE         Dots-per-inch (DPI) for output images. Default is 600.  [x>=0]
  -F, --out-format [png|pdf|svg|jpg]
                                  Output format for images. Default is ``png`` Valid options are:
                                  ['png', 'pdf', 'svg', 'jpg'].
  -D, --drop-legend               Drop legend from map. Legend is shown by default.
  --help                          Show this message and exit.
```

This command intentionally limits the options available to the user because it is meant to produce standard maps that are commonly desired for any supply curve. The main changes that the user can make are to change the DPI of the output image (e.g., for less detaild/smaller image file sizes, set to 300) and to provide a custom `--boundaries` vector dataset. The latter option merits some additional explanation.

By default, the maps will be generated using a boundaries file consisting of state boundaries for CONUS at 1:50m scale. This is intended to be suitable for the majority use case for reV: supply curves for all of CONUS. For other regions, or a subset of regions, users should provide a different boundaries file, suitable to their region. This should be a polygon vector dataset in one of the GIS formats that can be read by `fiona`/`geopandas`.  It will be used to add boundary lines to the maps (on top of the supply curve points) and also to create a drop-shadow for the region shown in the map.

For more customizable maps, refer to the next section.

## Mapping a Column from a Supply Curve
The `map-column` command can be used to generate a single map for any one column from the input supply curve. The output map follows a style similar to `make-maps`, but the user has greater flexibility not only over which column to map, but also some of the map formatting options, including colors, legend breaks, and legend title.

This command can be run according to the following usage:
```commandline
Usage: map-column [OPTIONS]

  Generates a single map from an input supply curve for the specified column,
  with basic options for formatting.

Options:
  -i, --supply_curve_csv FILE  Path to supply curve CSV file.  [required]
  -o, --out_folder DIRECTORY   Path to output folder for maps.  [required]
  -c, --column TEXT            Column to map  [required]
  -C, --colormap TEXT          Color map to use for the column. Refer to https
                               ://matplotlib.org/stable/tutorials/colors/color
                               maps.html for valid options. If not specified,
                               the viridis colormap will be applied.
  -T, --legend_title TEXT      Title to use for the map legend. If not
                               provided, legend title will be the column name
  -B, --legend_breaks TEXT     Breaks to use for the map legend. Should be
                               formatted like a list, e.g. : "[10, 50, 100,
                               150]". If not provided, a 5-class quantile
                               classification will be used to derive the
                               breaks.
  -b, --boundaries FILE        Path to vector dataset with the boundaries to
                               map. Default is to use state boundaries for
                               CONUS from Natural Earth (1:50m scale), which
                               is suitable for CONUS supply curves. For other
                               region, it is recommended to provide a more
                               appropriate boundaries dataset.
  -d, --dpi INTEGER RANGE      Dots-per-inch (DPI) for output images. Default
                               is 600.  [x>=0]
  --help                       Show this message and exit.
```

This command is intended to give the user options not available in the more standardized `make-maps` command, including:
1. Ability to map additional/other columns
2. More control over the output map appearance for any column, including the columns included in `make-maps`

For even more flexibility in generating maps, it is recommended to use the `reView.utils.plots.map_geodataframe_column` function in their own script(s). In combination with some basic data preparation with `geopandas`, this function can be used to map any column from either a point or polygon GeoDataFrame, with lots of flexibility around formatting both the map and the other elements like the title and legend. Below are two examples of using `reView.utils.plots.map_geodataframe_column()` for point and polygon data, respectively.

Example 1: Mapping Supply Curve Points
```python
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from reView.utils.plots import map_geodataframe_column

# inputs
supply_curve_path = Path("/path/to/supply-curve.csv")
state_boundaries_path = Path("/path/to/states.gpkg")

# load supply curve and convert to geodataframe
supply_curve_df = pd.read_csv(supply_curve_path)
supply_curve_gdf = gpd.GeoDataFrame(
    supply_curve_df,
    geometry=gpd.points_from_xy(
        x=supply_curve_df['longitude'], y=supply_curve_df['latitude']
    ),
    crs="EPSG:4326"
)

# load states (to be used as the boundary layer)
states_gdf = gpd.read_file(state_boundaries_path)

# specify mapping parameters
col_name = "capacity"
color_map = "GnBu"
breaks = [500, 1000, 1500, 2000]
map_extent = states_gdf.buffer(0.05).total_bounds

# create the map
g = map_geodataframe_column(
    supply_curve_gdf,
    col_name,
    color_map=color_map,
    breaks=breaks,
    map_title="Styling Map",
    legend_title=col_name.title(),
    background_df=background_gdf,
    boundaries_df=states_gdf,
    extent=map_extent,
    # change the way the points will be displayed:
    #   set their size bigger, remove the outline,
    #   and set the marker to a circle
    layer_kwargs={
      "s": 4,
      "linewidth": 0,
      "marker": "o"
    },
    # change the legend display:
    #   change markers/patches to circles, turn the frame on,
    #   frame and position on the right side of the map,
    #   at the bottom
    legend_kwargs={
        "marker": "o",
        "frameon": True,
        "bbox_to_anchor": (1, 0),
        "loc": "upper left"
    }
)
# remove extra padding in the figure
plt.tight_layout()

# save map as a png file
out_png_name = "points_map.png"
out_png = Path("/path/to/output/maps").joinpath(out_png_name)
g.figure.savefig(out_png, dpi=600)
plt.close(g.figure)
```

Example 2: Aggregating and Mapping Supply Curve to Polygons
```python
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from reView.utils.plots import map_geodataframe_column

# inputs
supply_curve_path = Path("/path/to/supply-curve.csv")
county_boundaries_path = Path("/path/to/counties.gpkg")
state_boundaries_path = Path("/path/to/states.gpkg")

# load supply curve and convert to geodataframe
supply_curve_df = pd.read_csv(supply_curve_path)
supply_curve_gdf = gpd.GeoDataFrame(
    supply_curve_df,
    geometry=gpd.points_from_xy(
        x=supply_curve_df['longitude'], y=supply_curve_df['latitude']
    ),
    crs="EPSG:4326"
)

# load counties (to be used as the mapping layer)
counties_gdf = gpd.read_file(county_boundaries_path)

# load states (to be used as the boundary layer)
states_gdf = gpd.read_file(state_boundaries_path)

# get aggregate capacity for each county from the supply curve
county_capacity_df = supply_curve_gdf.groupby(
    "cnty_fips"
)["capacity"].sum().reset_index()

# join county capacity to county geodataframe
county_capacity_gdf = counties_gdf.merge(
    county_capacity_df, how="inner", on="cnty_fips"
)

# specify mapping parameters
col_name = "capacity"
color_map = "YlOrRd"
breaks = [5000, 10000, 15000, 20000]
map_extent = county_background_gdf.buffer(0.05).total_bounds

# create the map
g = map_geodataframe_column(
    county_capacity_gdf,
    col_name,
    color_map=color_map,
    breaks=breaks,
    map_title="Polygons Map",
    legend_title=col_name.title(),
    background_df=county_background_gdf,
    boundaries_df=states_gdf,
    extent=map_extent,
    # change the way the polygons will be displayed:
    #   use narrow, gray edges
    layer_kwargs={"edgecolor": "gray", "linewidth": 0.5},
    # change the way the boundaries will be displayed:
    #   layer them on top of the counties with
    #   thicker, black edges
    boundaries_kwargs={
        "linewidth": 1,
        "zorder": 2,
        "edgecolor": "black",
    },
    # change the legend display:
    #   change markers/patches to squares, turn off the
    #   frame and position on the right side of the map,
    #   in the center
    legend_kwargs={
        "marker": "s",
        "frameon": False,
        "bbox_to_anchor": (1, 0.5),
        "loc": "center left"
    }
)
# remove extra padding in the figure
plt.tight_layout()

# save map as a png file
out_png_name = "polygons_map.png"
out_png = Path("/path/to/output/maps").joinpath(out_png_name)
g.figure.savefig(out_png, dpi=600)
plt.close(g.figure)
```

## Unpacking Characterizations
The `unpack-characterizations` command can be used to unpack one or more "characterization" columns from a supply curve CSV. These columns typically contain a summary of land-use or other spatial information that characterizes the developable land within each supply curve project site.

The data in these columns are encoded as JSON strings, and thus, not easily accessible for further analysis. Unpacking these JSON strings into useable data is both complicated and slow. The `unpack-characterizations` tool was developed to make this process easier.

An example of a characterization column would be `fed_land_owner`, and a single value in this column could have a value like:
```json
{"2.0": 10.88888931274414, "6.0": 20.11111068725586, "255.0": 2604.860107421875}
```
This JSON string tells us the count of grid cells corresponding to different federal land owners (USFS, BLM, and Non-Federal, respectively) within the developable land for that supply curve project site. Using `unpack-characterizations`, we can unpack this data to give us each of these values in a new, dedicated column, converted to square kilometers:
- `BLM_area_sq_km`: `0.162899997`
- `FS_area_sq_km`: `0.088200003`
- `Non-Federal_area_sq_km`: `21.09936687`

Usage of this command is as follows:
```commandline
Usage: unpack-characterizations [OPTIONS]

  Unpacks characterization data from the input supply curve dataframe,
  converting values from embedded JSON strings to new standalone columns, and
  saves out a new version of the supply curve with these columns included.

Options:
  -i, --supply_curve_csv FILE  Path to bespoke wind supply curve CSV file
                               created by reV  [required]
  -m, --char_map FILE          Path to JSON file storing characterization map
                               [required]
  -o, --out_csv FILE           Path to CSV to store results  [required]
  -c, --cell_size FLOAT        (Optional) Cell size in meters of
                               characterization layers. Default is 90.
  --overwrite                  Overwrite output CSV if it already exists.
                               Default is False.
  --help                       Show this message and exit.
```

The trickiest part of using `unpack-characterizations` is defining the correct "characterization map" to specify in `-m/--char_map`. This should be a JSON file that defines how to unpack and recast values from the characterization JSON strings to new columns.

Each top-level key in this JSON should be the name of a column of `-i/--supply_curve_csv` containing characterization JSON data. Only the columns you want to unpack need to be included.

The corresponding value should be a dictionary with the following keys: `method`, `recast`, and `lkup` OR `rename`. Details for each are provided below:
- `method`: Must be one of `category`, `sum`, `mean`, or `null`. Note: These correspond to the `method` used for the corresponding layer in the `data_layers` input to reV supply-curve aggregation configuration.
- `recast`: Must be one of `area` or None. This defines how values in the JSON will be recast to new columns. If `area` is specified, they will be converted to area values. If null, they will not be changed and will be passed through as-is.
- `lkup`: This is a dictionary for remapping categories to new column names. Using the `fed_land_owner` example above, it would be: `{"2.0": "FS", "6.0": "BLM", "255.0": "Non-Federal"}`. This follows the same format one could use for ``pandas.rename(columns=lkup)``. This parameter should be used when `method` = `category`. It can also be specified as `null` to skip unpacking of the column.
- `rename`: This is a string indicating what name to use for the new column. This should be used when `method` != `category`.

A valid example of a characterization map can be found [here](tests/data/characterization-map.json) in the test data.

## Unpacking Turbines
The `unpack-turbines` command can be used to convert a "bespoke" wind supply curve created by reV to include a point location for each individual turbine in the supply curve. This command can be used according to the following usage:

```commandline
Usage: unpack-turbines [OPTIONS]

  Unpack individual turbines from each reV project site in a reV supply curve
  CSV, produced using "bespoke" (i.e., SROM) turbine placement.

Options:
  -i, --supply_curve_csv PATH  Path to bespoke wind supply curve CSV file
                               created by reV  [required]
  -o, --out_gpkg PATH          Path to regions shapefile containing labeled
                               geometries  [required]
  -n, --n_workers INTEGER      Number of workers to use for parallel
                               processing.Default is 1 which will unpack
                               turbines from each supply curve grid cell in
                               parallel. This will be slow. It is recommended
                               to use at least 4 workers if possible
                               [default: 1]
  --overwrite                  Overwrite output geopackage if it already
                               exists. Default is False.
  --help                       Show this message and exit.
```

Turbine locations from the input supply curve CSV will be unpacked and returned as a GeoPackage. If you are unsure whether your supply curve includes individual turbine locations, look for the `turbine_x_coords` and `turbine_y_coords` columns, which should be present. By default, this command runs in serial (on one worker), which can be slow. For most uses, it is recommended to use parallel processing with >=4 workers.
