import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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


def plot_distance_heatmap(
        dist_df: pd.DataFrame,
        vmax_quantile: float = 0.99,
        cmap: str = "viridis",
        ax=None,
        title: str = "Wasserstein distance between daily H0 diagrams",
):
    """
    Heatmap of the distance matrix with dates on both axes.
    The color scale is capped at a quantile so a few extreme days don't wash out the rest.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    values = dist_df.to_numpy()
    n = len(values)
    vmax = np.quantile(values[np.triu_indices(n, k=1)], vmax_quantile)

    t = mdates.date2num(dist_df.index.to_pydatetime())
    # pcolormesh uses the real dates as coordinates, so missing days don't distort the axes
    mesh = ax.pcolormesh(
        t, t, values, shading="nearest", cmap=cmap, vmin=0, vmax=vmax, rasterized=True
    )

    for axis in (ax.xaxis, ax.yaxis):
        locator = mdates.AutoDateLocator()
        axis.set_major_locator(locator)
        axis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.invert_yaxis()          # matrix convention: first day at the top-left
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="distance")
    return fig, ax


if __name__ == "__main__":

    # example
    dist_df = pd.read_csv(
        "data/wasserstein_distance_matrix.csv", index_col=0, parse_dates=True
    )
    plot_distance_heatmap(dist_df, vmax_quantile=0.99, cmap="viridis")
    plt.show()