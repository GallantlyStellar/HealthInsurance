#!/usr/bin/env python3
# coding: utf-8

"""
Assignment for Portfolio.

Regression analysis of health insurance information.

GallantlyStellar
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from scipy.stats import false_discovery_control
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
# plt.close("all")


# Split training and testing data
X_train, X_test, y_train, y_test = splitData(encoded)

# Fit the model
lr, preds = linearModelSM(X_train, y_train, X_test)

lrFindings = (
    pd.DataFrame(
        data={
            "coefficient": lr.params,
            # correct p-values for false discovery rate with Benjamini-Hochberg
            "adjusted p-value": false_discovery_control(lr.pvalues, method="bh"),
        }
    )
    .round(6)
    .sort_values("adjusted p-value")
)


pca = pca(encoded.drop("charges", axis=1))
pcaFindings = pd.concat(
    [
        # DF of PC eigenvalue ratios
        pd.DataFrame(
            {"explained_variance_ratio": pca.explained_variance_ratio_},
            index=[f"PC{i+1}" for i in range(pca.n_components_)],
        ).T,
        # DF of PC eigenvector components
        pd.DataFrame(
            pca.components_,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            index=encoded.drop("charges", axis=1).columns,
        ),
    ]
)

pcs = pca.transform(encoded.drop("charges", axis=1))

# Visualize PCs compared to certain influential features
fig = px.scatter_3d(
    x=pcs[:, 0],
    y=pcs[:, 1],
    z=df["charges"],
    color=df["smoker"].replace({0: "Nonsmoker", 1: "Smoker"}),
    labels={
        "x": "Principal Component 1",
        "y": "Principal Component 2",
        "z": "Charges (USD)",
        "color": "Smoker Status",
        "size": "Age",
    },
    opacity=0.8,
    title="Dimensionalty-Reduced Features vs Charges",
)
fig = fig.update_traces(marker=dict(size=6))
fig.write_html("../figures/pcs.html")

# Evaluate the model
print(f"Test set RMSE: {root_mean_squared_error(y_test, preds)}")
print(f"Test set R^2: {r2_score(y_test, preds)}")
print(lr.summary2())

# Summarize findings
print(pcaFindings)
print(lrFindings)
