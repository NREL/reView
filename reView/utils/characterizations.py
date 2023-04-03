# -*- coding: utf-8 -*-
"""Methods for unpacking characterization data from supply curves so that the
data can be accessed as proper columns rather than embedded JSON data.

@author: Mike Gleason
"""
import json
import warnings

import pandas as pd
import tqdm


# pylint: disable=raise-missing-from
def recast_categories(df, col, lkup, cell_size_sq_km):
    """
    Recast the JSON string found in ``df[col]`` to new columns in the
    dataframe. Each element in the embedded JSON strings will become a new
    column following the casting specified by ``lkup`` and ``cell_size_sq_km``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input pandas dataframe
    col : str
        Name of column in df containing embedded JSON values
        (e.g., ``"{'0': 44.3, '1': 3.7}"``).
    lkup : dict
        Dictionary used to map keys in the JSON strings to new, more meaningful
        names. Following the example above, this might be
        ``{"0": "Grassland", "1": "Water"}``.This follows the same format one
        could use for ``pandas.rename(columns=lkup)``.
    cell_size_sq_km : [int, None]
        Optional value indicating the cell size of the characterization data
        being recast.

        If specified, it has two effects. First, it will be used to convert
        values of the JSON to values of area in units of square kilometers
        during the recast process. Second, all recast column names specified in
        ``lkup`` will have the suffix `_area_sq_km` added to them. Continuing
        from the examples above, if ``cell_size_sq_km=0.0081``, the value
        `44.3` above would be multipled by `0.0081`, producing a new value of
        `0.35883`. This value would be stored in a new column named
        ``"Water_area_sq_km"``.

        If not specified, which is the default, no conversion to area will be
        applied, values from the JSON will be passed through (or filled with
        ``0`` if missing), and column names specified in ``lkup`` will be used
        verbatim in the output dataframe.

    Returns
    -------
    pandas.DataFrame
        New pandas dataframe with additional recast columns appended to the
        input dataframe.

    Raises
    ------
    TypeError
        A TypeError will be raised if one or more values in ``df[col]`` is not
        a str dtype.
    """

    try:
        elements = ','.join(df[col].tolist())
    except TypeError:
        raise TypeError(
            f"Unable to recast column {col} to categories. "
            "Some values are not str."
        )
    col_data = json.loads(f"[{elements}]")
    col_df = pd.DataFrame(col_data)
    col_df.fillna(0, inplace=True)
    col_df.drop(
        columns=[c for c in col_df.columns if c not in lkup.keys()],
        inplace=True
    )
    col_df.rename(columns=lkup, inplace=True)
    if cell_size_sq_km is not None:
        col_df *= cell_size_sq_km
        col_df.rename(
            columns={c: f"{c}_area_sq_km" for c in col_df.columns},
            inplace=True
        )

    col_df.index = df.index

    out_df = pd.concat([df, col_df], axis=1)

    return out_df


# pylint: disable=too-many-branches
def unpack_characterizations(  # noqa: C901
        in_df, characterization_remapper, cell_size_m=90
):
    """
    Unpacks characterization data from the input supply curve dataframe,
    converting values from embedded JSON strings to new standalone columns.

    Parameters
    ----------
    in_df : pandas.DataFrame
        Dataframe to be unpacked. Typically, this is a DataFrame loaded
        from a reV supply curve CSV file.
    characterization_remapper : dict
        This dictionary defines how to unpack and recast values from the
        characterization JSON strings to new columns.

        Each top-level key in this dictionary should be the name of a
        column of ``in_df`` containing characterization JSON data. Only the
        columns you want to unpack need to be included.

        The corresponding value should be a dictionary with the following keys:
        "method", "recast", and "lkup" OR "rename".
        Details for each are provided below:
        - "method": Must be one of "category", "sum", "mean", or None.
            Note: These correspond to the "method" used for the
            corresponding layer in the "data_layers" input
            to reV supply-curve aggregation configuration.
        - "recast": Must be one of "area" or None. This defines how values
            in the JSON will be recast to new columns. If "area" is specified,
            they will be converted to area values. If null, they will not be
            changed and will be passed through as-is.
        - "lkup": This is a dictionary for remapping categories to new
            column names (see documentation of recast_categories() for more
             information). It should be used when "method" = "category".
             It can also be specified as null to skip unpacking of the column.
        - "rename": This is a string indicating what name to use for the new
            column. This should be used when "method" != "category".

         A valid example for this parameter can be loaded from
         ``tests/data/characterization-map.json``.
    cell_size_m : int
        Optional cell size of the characterization layers used in reV.
        Default is the current standard value of 90m. This value is necessary
        if you

    Returns
    -------
    pandas.DataFrame
        New pandas dataframe with additional columns for data unpacked
        from characterization columns.

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """

    cell_size_sq_km = cell_size_m**2 / 1e6

    validate_characterization_remapper(characterization_remapper, in_df)

    for char_col, col_remapper in tqdm.tqdm(
        characterization_remapper.items(),
        desc="Unpacking Characterizations"
    ):
        method = col_remapper.get("method", None)
        recast = col_remapper.get("recast", None)
        lkup = col_remapper.get("lkup", None)
        rename = col_remapper.get("rename", char_col)

        if method == "category":
            if lkup is None:
                warnings.warn(f"Skipping {char_col}: No lkup provided")
            else:
                if recast == "area":
                    in_df = recast_categories(
                        in_df, char_col, lkup, cell_size_sq_km
                    )
                elif recast is None:
                    in_df = recast_categories(in_df, char_col, lkup, None)
        elif method == "sum":
            if recast == "area":
                in_df[f"{rename}_area_sq_km"] = (
                    in_df[char_col] * cell_size_sq_km
                )
            elif recast is None:
                if rename != char_col:
                    in_df[rename] = in_df[char_col]
        elif method == "mean":
            if recast == "area":
                in_df[f"{rename}_area_sq_km"] = (
                    in_df[char_col] * in_df["area_sq_km"]
                )
            elif recast is None:
                if rename != char_col:
                    in_df[rename] = in_df[char_col]
        elif method is None:
            warnings.warn(f"Skipping {char_col}: No method provided")

        else:
            raise ValueError(f"Invalid method: {method}")

        in_df = in_df.copy()

    return in_df


def validate_characterization_remapper(
    characterization_remapper, supply_curve_df
):
    """
    Ensure the validity of the input characterization map. Intended for use as
    a helper function to unpack_characterizations()

    Parameters
    ----------
    characterization_remapper : dict
        This dictionary defines how to unpack and recast values from the
        characterization JSON strings to new columns. See documentation of
        unpack_characterizations() for details.
    supply_curve_df : list
        DataFrame that will be used with characterization_remapper.

    Raises
    ------
    KeyError
        A KeyError will be raised if any of the input column names in
        characterization_remapper are not present in supply_curve_df.
    ValueError
        A ValueError will be raised if any invalid combinations of
        parameters are encountered in characterization_remapper.
    """

    characterization_cols = list(characterization_remapper.keys())
    df_cols = supply_curve_df.columns.tolist()
    cols_not_in_df = list(set(characterization_cols).difference(set(df_cols)))
    if len(cols_not_in_df) > 0:
        raise KeyError(
            "Invalid column name(s) in characterization_remapper. "
            "The following column name(s) were not found in the input "
            f"dataframe: {cols_not_in_df}."
        )

    for col_name, col_remapper in characterization_remapper.items():
        method = col_remapper.get("method", None)
        recast = col_remapper.get("recast", None)
        lkup = col_remapper.get("lkup", None)
        rename = col_remapper.get("rename", None)

        valid_methods = ("category", "sum", "mean", None)
        if method not in valid_methods:
            raise ValueError(
                f"{col_name} - Invalid value for method: {method}."
                f"Must be one of {valid_methods}."
            )

        valid_recasts = ("area", None)
        if recast not in valid_recasts:
            raise ValueError(
                f"{col_name} - Invalid value for recast: {recast}."
                f"Must be one of {valid_recasts}."
            )

        if method == "category":
            if lkup is not None and not isinstance(lkup, dict):
                raise ValueError(
                    f"{col_name} - Invalid value for lkup: {lkup}. "
                    f"Must be a dict or None when method={method}."
                )
            if rename is not None:
                raise ValueError(
                    f"{col_name} - Invalid value for rename: {rename}."
                    f"Must be None when method={method}."
                )
        elif method in ("sum", "mean"):
            if lkup is not None:
                raise ValueError(
                    f"{col_name} - Invalid value for lkup: {lkup}. "
                    f"Must be None when method={method}."
                )
            if rename is not None and not isinstance(rename, str):
                raise ValueError(
                    f"{col_name} - Invalid value for rename: {rename}. "
                    f"Must be None or a string when method={method}."
                )
        elif method is None:
            if lkup is not None:
                raise ValueError(
                    f"{col_name} - Invalid value for lkup: {lkup}. "
                    f"Must be None when method={method}."
                )
            if recast is not None:
                raise ValueError(
                    f"{col_name} - Invalid value for recast: {recast}. "
                    f"Must be None when method={method}."
                )
            if rename is not None:
                raise ValueError(
                    f"{col_name} - Invalid value for rename: {rename}. "
                    f"Must be None when method={method}."
                )
