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
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error

from eda import ageChargeScatterplots
from importAndClean import importDF
from linearModel import linearModel, oneHot, pca, splitData
from visualize import bivariate, univariate

# Make plot fonts bigger
plt.rcParams.update({"font.size": 20})

# read in data
df = importDF(path="../assets/data/raw/insurance.csv")
univariate(df, 3, 3)
bivariate(df, "charges", 2, 3)
# plt.show()
plt.close("all")

ageChargeScatterplots(df)
# plt.show()
plt.close("all")

encoded = oneHot(df)
X_train, X_test, y_train, y_test = splitData(encoded)
lr = linearModel(X_train, y_train)
root_mean_squared_error(y_test, lr.predict(X_test))
r2_score(y_test, lr.predict(X_test))

pca = pca(encoded)
# pca.explained_variance_ratio_
pcs = pca.transform(encoded.drop("charges", axis=1))
# TODO: add line

fig = px.scatter_3d(
    x=pcs[:, 0],
    y=pcs[:, 1],
    z=df["charges"],
    color=df["smoker"],
    labels={"x": "PC1", "y": "PC2", "z": "Charges (USD)"},
)

fig.write_html("../report/figures/pcs.html")
