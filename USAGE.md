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
  unpack-characterizations  Unpacks characterization data from the input supply curve dataframe, converting values from embedded JSON strings to new standalone columns, and saves out a new version of the supply curve with these columns included.
  unpack-turbines           Unpack individual turbines from each reV project site in a reV supply curve CSV, produced using "bespoke" (i.e., SROM) turbine placement.
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
