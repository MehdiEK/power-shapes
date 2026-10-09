import matplotlib.pyplot as plt


def plot_persistence_diagram(diagram, net_load):
    """
    Plots the persistence diagram for H0.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(diagram[:, 0], diagram[:, 1])
    vmin, vmax = min(net_load), max(net_load)
    plt.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5)
    plt.xlabel("Naissance")
    plt.ylabel("Mort")
    plt.title("Diagramme de persistance H0 — creux du net load")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
