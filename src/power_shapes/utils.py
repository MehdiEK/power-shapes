from collections.abc import Mapping, Sequence

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


def scale_columns_by_daily_mean(
    df: pd.DataFrame,
    columns_by_mean: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Divide each mapped value column by its key column's daily mean.

    Daily groups use the calendar date in ``datetime``. The input dataframe is
    not modified; the scaled dataframe is returned.
    """
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain a 'datetime' column")
    # Grouping by calendar date only is intentional; timestamps within a day
    # should share the same normalization factor.
    if not is_datetime64_any_dtype(df["datetime"].dtype):
        raise TypeError("'datetime' column must have a datetime dtype")

    # Validate every column referenced by the mapping before doing any work.
    referenced_columns = {
        column
        for mean_column, value_columns in columns_by_mean.items()
        for column in (mean_column, *value_columns)
    }
    missing_columns = referenced_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(f"Columns not found in DataFrame: {sorted(missing_columns)}")

    non_numeric_columns = sorted(
        column
        for column in referenced_columns
        if not is_numeric_dtype(df[column].dtype)
    )
    if non_numeric_columns:
        raise TypeError(
            f"Columns used for scaling must be numeric: {non_numeric_columns}"
        )

    # Work on a copy so the caller's dataframe remains unchanged.
    result = df.copy()
    dates = df["datetime"].dt.date

    for mean_column, value_columns in columns_by_mean.items():
        # transform broadcasts each date's mean back to its original rows.
        daily_means = df[mean_column].groupby(dates).transform("mean")
        if daily_means.eq(0).any():
            raise ValueError(
                f"Cannot scale columns by '{mean_column}': "
                "its daily mean is zero"
            )

        for value_column in value_columns:
            # Dividing each observation by the aligned daily mean scales it
            # relative to the corresponding key column's average for that day.
            result[value_column] = df[value_column].div(daily_means).array

    return result
