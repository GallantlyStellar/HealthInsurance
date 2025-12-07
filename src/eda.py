#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for health insurance portfolio project.

Explorations of health insurance data.

GallantlyStellar
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ageChargeScatterplots(df: pd.DataFrame) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.scatterplot(
        x=df["age"],
        y=df["charges"],
        hue=df["bmi"] > 30,
        edgecolor=None,
        ax=ax1,
    )
    ax1.spines[["right", "top"]].set_visible(False)
    ax1.set_xlabel("Age (years)")
    ax1.set_ylabel("Charges (USD)")
    ax1.legend(title="Obese (BMI>=30)", loc="lower right")
    ax1.set_title("Age vs Charges by obesity classification")

    with pd.option_context("future.no_silent_downcasting", True):
        sns.scatterplot(
            x=df["age"],
            y=df["charges"],
            hue=df["smoker"].replace({0: False, 1: True}).astype("bool"),
            edgecolor=None,
            ax=ax2,
        )
        ax2.spines[["right", "top"]].set_visible(False)
        ax2.set_xlabel("Age (years)")
        ax2.set_ylabel("Charges (USD)")
        ax2.legend(title="Smoker", loc="lower right")
        ax2.set_title("Age vs Charges by smoker classification")
    return fig
