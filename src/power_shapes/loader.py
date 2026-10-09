import pandas as pd 
from pathlib import Path


def load_rte_data(
    file_path:Path = Path('./data/donnees-rte.csv')
):
    """
    Enales RTE data loading and preprocessing from the csv file 
    stored in data.
    """
    # load data from the csv file
    df = pd.read_csv(file_path, sep=';')

    # convert datetime column to datetime type
    df["datetime"] = pd.to_datetime(
        df["Date et Heure"], utc=True
    )
    df.drop(columns=["Date et Heure"], inplace=True)

    # drop useless information and rename columns
    df = df[
        [
            'datetime', 
            'Consommation (MW)', 
            'Solaire (MW)', 
            'Eolien (MW)'
        ]
    ]
    df = df.rename(
        columns={
            'Consommation (MW)': 'consumption_mw',
            'Solaire (MW)': 'solar_mw',
            'Eolien (MW)': 'wind_mw'
        }
    )

    # drop nana values of any kind
    df.dropna(
        subset=[
            "consumption_mw", 
            "solar_mw", 
            "wind_mw"
        ], 
        inplace=True
    )

    # get net demand solar 
    df['net_demand_solar_mw'] = df['consumption_mw'] - df['solar_mw']
    df['net_demand_mw'] = (
        df['consumption_mw'] 
        - df['solar_mw']
        - df['wind_mw']
    )

    return df


if __name__ == '__main__':
    df = load_rte_data('tests/fixtures/samples.csv')
    print(df)


