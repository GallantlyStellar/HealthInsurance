#!/usr/bin/env python3
# coding: utf-8

"""
Assignment for Portfolio.

Regression analysis of health insurance information.

GallantlyStellar
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error

from eda import ageChargeScatterplots
from importAndClean import importDF
from linearModel import linearModelSM, oneHot, pca, splitData
from visualize import bivariate, corr, univariate

# Make plot fonts bigger
plt.rcParams.update({"font.size": 20})

# read in data
df = importDF(path="../assets/data/raw/insurance.csv")

# EDA
univariate(df, 3, 3)
bivariate(df, "charges", 2, 3)
ageChargeScatterplots(df)
encoded = oneHot(df)
corr(encoded)
plt.close("all")


# Split training and testing data
X_train, X_test, y_train, y_test = splitData(encoded)

# Fit the model
lr, preds = linearModelSM(X_train, y_train, X_test)

# Evaluate the model
root_mean_squared_error(y_test, preds)
r2_score(y_test, preds)
lr.summary2()


pca = pca(encoded.drop("charges", axis=1))
pca.explained_variance_ratio_
pcs = pca.transform(encoded.drop("charges", axis=1))

# Visualize PCs compared to certain influential features
fig = px.scatter_3d(
    x=pcs[:, 0],
    y=pcs[:, 1],
    z=df["charges"],
    symbol=(df["bmi"] > 30).replace({False: "Not Obese", True: "Obese"}),
    color=df["smoker"].replace({0: "Nonsmoker", 1: "Smoker"}),
    labels={
        "x": "Principal Component 1",
        "y": "Principal Component 2",
        "z": "Charges (USD)",
        "color": "Color: Smoker Status",
        "symbol": "Symbol: BMI>30",
    },
    opacity=0.8,
    title="Dimensionalty-Reduced Features vs Charges",
)
fig = fig.update_traces(marker=dict(size=6))
fig.write_html("../report/figures/pcs.html")
