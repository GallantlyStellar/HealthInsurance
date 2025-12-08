#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for health insurance portfolio project.

Create multi-plot visualizations of source data.

GallantlyStellar
"""

import pandas as pd
from matplotlib.figure import Figure


def univariate(df: pd.DataFrame, n_rows: int = 4, n_cols: int = 4) -> Figure:
    """
    Create univariate visualizations of a df.

    Args:
        df (DataFrame): A Pandas DataFrame to visualize.
        n_rows (int): The number of rows to display in the final figure.
        n_cols (int): The number of columns to display in the final figure.

    Returns:
        fig (matplotlib Figure): Use plt.show() to access figure.

    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert n_rows * n_cols >= len(df.columns), "Not enough plots for all cols"

    plot_count = len(df.columns)
    n_drop = (n_rows * n_cols) - plot_count

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    axs = axs.reshape(-1)  # flatten to iterate
    # remove 2 unneeded plots
    _ = [axs[i].remove() for i in range(-n_drop, 0)]

    fig.supylabel("Occurences", x=-0.0005)
    fig.supxlabel("Feature Value", y=-0.003)
    cmap = plt.get_cmap("tab20", plot_count)
    color = [cmap(i) for i in range(0, plot_count)]
    np.random.seed(6396)
    np.random.shuffle(color)
    for i in range(0, plot_count):
        ax = axs[i]
        ax.set_title(df.columns[i])
        ax.spines[["right", "top"]].set_visible(False)
        col = df.iloc[:, i]
        colUniqueVals = col.drop_duplicates().sort_values()
        if col.nunique() > 10:  # continuous values
            ax.hist(col, color=color[i])
        # binary values
        elif colUniqueVals.dropna().isin([0, 1]).all():
            col = col.astype(str).replace(
                {
                    "0": "False",
                    "1": "True",
                    "<NA>": "NA",
                    "nan": "NA",
                }
            )
            val_counts = col.value_counts().sort_index()
            ax.bar(val_counts.index, val_counts.values, color=color[i])
            del val_counts
        # discrete, non-binary values
        else:
            col = col.astype(str).replace(
                {
                    "<NA>": "NA",
                    "nan": "NA",
                }
            )
            val_counts = col.value_counts().sort_index()
            labels = val_counts.index.astype(str)
            ax.bar(labels, val_counts.values, color=color[i])
            ax.set_xticks(range(len(labels)))  # suppress warning
            ax.set_xticklabels([label[:5] + "." if len(label) > 5 else label for label in labels])
            del val_counts, labels
        del col, colUniqueVals
    return fig


def bivariate(df: pd.DataFrame, target: str, n_rows: int = 4, n_cols: int = 4) -> Figure:
    """
    Create bivariate visualizations of a df.

    Args:
        df (DataFrame): A Pandas DataFrame to visualize.
        target (str): String matching the col name to use as the response var.
        n_rows (int): The number of rows to display in the final figure.
        n_cols (int): The number of columns to display in the final figure.

    Returns:
        fig (matplotlib Figure): Use plt.show() to access figure.

    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    assert n_rows * n_cols >= (len(df.columns) - 1), "Not enough plots for all cols"
    plot_count = len(df.columns) - 1  # don't plot target vs itself
    n_drop = (n_rows * n_cols) - plot_count
    target_nunique = df[target].nunique()

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    axs = axs.reshape(-1)  # flatten to iterate
    # remove 2 unneeded plots
    _ = [axs[i].remove() for i in range(-n_drop, 0)]

    for i in range(0, plot_count):
        ax = axs[i]
        ax.set_title(df.columns[i])
        ax.spines[["right", "top"]].set_visible(False)
        col = df.iloc[:, i]
        colUniqueVals = col.drop_duplicates().sort_values()
        # if target is continuous (regressions),
        # use stacked hists or scatterplots
        if target_nunique > 10:
            # continuous values
            if col.nunique() > 10:
                sns.scatterplot(
                    x=col,
                    y=df[target],
                    color=sns.color_palette("pastel")[0],
                    ax=ax,
                    edgecolor=None,
                )
                # discrete values (no special case for binary, target is cont)
            else:
                sns.histplot(
                    x=df[target],
                    hue=col,
                    multiple="stack",
                    element="step",
                    palette="pastel",
                    ax=ax,
                    edgecolor=None,
                )
        # if target is categorical (classifications),
        # use hist if continuous, barplot of occurences if discrete
        else:
            # continuous values
            if col.nunique() > 10:
                sns.histplot(
                    x=col,
                    hue=df[target],
                    multiple="stack",
                    element="step",
                    palette="pastel",
                    ax=ax,
                    edgecolor=None,
                )
            # binary values
            elif colUniqueVals.dropna().isin([0, 1]).all():
                col = col.astype(str).replace(
                    {
                        "0": "False",
                        "1": "True",
                        "<NA>": "NA",
                        "nan": "NA",
                    }
                )
                if "NA" in col.values:
                    sns.countplot(
                        x=col,
                        hue=df[target],
                        ax=ax,
                        palette="pastel",
                        order=["False", "True", "NA"],
                    )
                else:
                    sns.countplot(
                        x=col,
                        hue=df[target],
                        ax=ax,
                        palette="pastel",
                        order=["False", "True"],
                    )
            # discrete, non-binary values
            else:
                col = col.astype(str).replace(
                    {
                        "<NA>": "NA",
                        "nan": "NA",
                    }
                )
                sns.countplot(x=col, hue=df[target], ax=ax, palette="pastel")
                labels = col.value_counts().sort_index().index.astype(str)
                ax.set_xticks(range(len(labels)))  # suppress warning
                ax.set_xticklabels(
                    [label[:5] + "." if len(label) > 5 else label for label in labels]
                )
                del labels
        # remove local legends and axis labels to have only one on figure
        # for discrete targets. Keep legends for continuous targets.
        if target_nunique < 10:
            ax.get_legend().remove()
            ax.set_xlabel("")
            ax.set_ylabel("")
        del col, colUniqueVals

    if target_nunique < 10:
        # get legend from last plot and use if target is categorical
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title=target, ncol=df[target].nunique(), loc="upper center")

        # Set X and Y labels for all plots
        fig.supylabel("Occurences", x=-0.0005)
        fig.supxlabel("Feature Value", y=-0.003)
    return fig


def corr(df: pd.DataFrame) -> Figure:
    """
    Create a heatmap of correlation coefficients.

    Args:
        df (DataFrame): A Pandas DataFrame to visualize.

    Returns:
        fig (matplotlib Figure): Use plt.show() to access figure.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    fig, ax = plt.subplots()
    ax = sns.heatmap(
        df.corr(),
        # triangle upper of true values to eliminate duplicated info
        mask=np.invert(np.triu(np.ones_like(df.corr(), dtype=bool))),
        # remove invert and use tril to eliminate autocorr diagonal
        # mask=np.tril(np.ones_like(df.corr(), dtype=bool)),
        annot=True,
        fmt="0.1f",  # 1 decimal on annotations
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=3,  # space between items
        square=True,
        xticklabels=True,  # show every label
        yticklabels=True,
        cbar_kws={"location": "left", "pad": 0.01},
    )
    ax.collections[0].colorbar.set_label("Correlation Coefficient", rotation=90, va="bottom")
    # Put x labels above
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-30, ha="right")
    # Put y labels on right
    ax.yaxis.tick_right()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="left")
    return fig
