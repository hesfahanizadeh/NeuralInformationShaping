"""Visualization Module"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_accs(
    accs: list,
    names: list,
    save_loc: str,
    linestyles=None,
    markers=None,
    legend: bool = False,
) -> None:
    """
    :param accs: Accuracy values
    :param names: Names of the algorithms
    :param save_loc: Location to save the plot
    :return: None
    """
    if linestyles is None:
        linestyles = ["--", "-", "-.", ":", "--", "-", "-."]

    if markers is None:
        markers = ["P", "o", "s", "D", "^", "v", "X"]
        
    # Create color palette
    sns.set_palette("husl", n_colors=len(accs))
    
    plt.close("all")
    _, ax = plt.subplots(figsize=(7, 7), dpi=600)

    for i, acc in enumerate(accs):
        # Convert acc numpy array to df 
        data = pd.DataFrame({
            'False Positive Rate': acc[0],
            'True Positive Rate': acc[1]
        })
        # data = pd.DataFrame(acc.reshape(-1, 1), columns=["Accuracy"])
        # data["Round"] = [i for i in range(acc.shape[1])] * acc.shape[0]
        sns.lineplot(
            data=data,
            x="False Positive Rate",
            y="True Positive Rate",
            label=names[i],
            # linestyle=linestyles[i],
            # marker=markers[i],
            linewidth=3,
            markersize=10,
        )

    plt.grid(linestyle="--", linewidth=2)
    plt.xlabel("False Positive Rate", fontweight="bold", fontsize=24)
    plt.ylabel("True Positive Rate", fontweight="bold", fontsize=24)

    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.linewidth"] = 2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(prop={"size": 20, "weight": "bold"})
    plt.tight_layout()
    if not legend:
        plt.legend([], [], frameon=False)
    plt.savefig(save_loc)
