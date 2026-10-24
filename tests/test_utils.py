import pandas as pd
import pandas.testing as pdt
import pytest

from power_shapes.common.utils import scale_columns_by_daily_mean


def test_scale_columns_by_daily_mean_groups_by_calendar_date():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 00:00"]
            ),
            "baseline": [2.0, 4.0, 8.0],
            "signal": [10.0, 20.0, 40.0],
            "other": [1.0, 2.0, 3.0],
        }
    )

    result = scale_columns_by_daily_mean(df, {"baseline": ["signal"]})

    assert result["signal"].tolist() == [10.0 / 3.0, 20.0 / 3.0, 5.0]
    pdt.assert_series_equal(result["other"], df["other"])
    pdt.assert_series_equal(df["signal"], pd.Series([10.0, 20.0, 40.0], name="signal"))


def test_scale_columns_by_daily_mean_requires_datetime_column():
    with pytest.raises(ValueError, match="'datetime'"):
        scale_columns_by_daily_mean(pd.DataFrame({"baseline": [1]}), {})


def test_scale_columns_by_daily_mean_requires_datetime_dtype():
    df = pd.DataFrame({"datetime": ["2024-01-01"], "baseline": [1]})

    with pytest.raises(TypeError, match="datetime dtype"):
        scale_columns_by_daily_mean(df, {})


def test_scale_columns_by_daily_mean_rejects_zero_daily_mean():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "baseline": [-1.0, 1.0],
            "signal": [2.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="daily mean is zero"):
        scale_columns_by_daily_mean(df, {"baseline": ["signal"]})
