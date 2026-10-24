from collections.abc import Mapping, Sequence

import numpy as np
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


def _offdiag(D):
    return D[np.triu_indices_from(D, k=1)]


def combine_distance_matrices(
        D_sub,
        D_sup,
        alpha=0.5,
        mode="sum",
        normalize=True
) -> np.array:
    """
    Combine sublevel and sup-level Wasserstein matrices (DataFrames or arrays, same index).

    mode : "sum" -> alpha*D_sub + (1-alpha)*D_sup   (L1 product metric)
           "l2"  -> sqrt(alpha*D_sub² + (1-alpha)*D_sup²)
           "max" -> max(D_sub, D_sup)
    normalize : divide each matrix by its median off-diagonal value first.
    """
    A, B = np.asarray(D_sub, float), np.asarray(D_sup, float)
    if normalize:
        A = A / np.median(_offdiag(A))
        B = B / np.median(_offdiag(B))

    if mode == "sum":
        C = alpha * A + (1 - alpha) * B
    elif mode == "l2":
        C = np.sqrt(alpha * A**2 + (1 - alpha) * B**2)
    elif mode == "max":
        C = np.maximum(A, B)
    else:
        raise ValueError("mode must be 'sum', 'l2' or 'max'")

    if isinstance(D_sub, pd.DataFrame):
        return pd.DataFrame(C, index=D_sub.index, columns=D_sub.columns)
    return C


def combine_distance_matrices_from_csv(
        sublevel_csv,
        superlevel_csv,
        alpha=0.5,
        mode="sum",
        normalize=True,
        save_output:str=None
) -> pd.DataFrame:
    """
    Combine sublevel and sup-level Wasserstein matrices (CSV files, same index).

    mode : "sum" -> alpha*D_sub + (1-alpha)*D_sup   (L1 product metric)
           "l2"  -> sqrt(alpha*D_sub² + (1-alpha)*D_sup²)
           "max" -> max(D_sub, D_sup)
    normalize : divide each matrix by its median off-diagonal value first.
    """
    D_sub = pd.read_csv(sublevel_csv, index_col=0, parse_dates=True)
    D_sup = pd.read_csv(superlevel_csv, index_col=0, parse_dates=True)

    combined_matrix = combine_distance_matrices(
        D_sub, D_sup,
        alpha=alpha,
        mode=mode,
        normalize=normalize
    )

    if save_output:
        combined_matrix.to_csv(save_output)

    return combined_matrix