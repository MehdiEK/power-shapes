import pandas as pd
import pandas.testing as pdt

from power_shapes.loader import load_rte_data


def test_loaded_data():
    df = load_rte_data('tests/fixtures/samples.csv')  # Replace with your actual loading function

    # 1. Assert the shape (rows, columns)
    assert df.shape == (5, 6)

    # 2. Assert the column names and order
    expected_columns = [
        "datetime", 
        "consumption_mw", 
        "solar_mw", 
        "wind_mw", 
        "net_demand_solar_mw", 
        "net_demand_mw"
    ]
    assert list(df.columns) == expected_columns

    # 3. Assert that timestamps are sampled every 30 minutes
    timestamps = pd.to_datetime(df["datetime"])
    intervals = timestamps.diff().dropna()

    expected_interval = pd.Timedelta(minutes=30)
    assert (intervals.abs() == expected_interval).all(), (
        "Timestamps are not consistently sampled every 30 minutes"
    )