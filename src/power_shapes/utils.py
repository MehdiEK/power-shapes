import pandas as pd
import matplotlib.pyplot as plt


def plot_series(day_ts:pd.DataFrame, title:str):
    """
    Plots net load, solar and wind production for a given day.
    """
    plt.figure(figsize=(10, 6))

    for col in day_ts.columns:
        if col != "datetime":
            plt.plot(day_ts["datetime"], day_ts[col], label=col)
    
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
