#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for health insurance portfolio project.

Test EDA functions. Paths assume tests
are run relative to the "src" directory.

GallantlyStellar
"""

import pandas as pd

from linearModel import linearModelSK, linearModelSM, oneHot, pca, splitData

df = pd.DataFrame(
    {
        "age": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "children": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "charges": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    }
)


def test_oneHot():
    # encoded = oneHot(df)
    encoded = oneHot(df)
    assert encoded.sum().sum(), "Data are not all numeric."


def test_splitData():
    X_train, X_test, y_train, y_test = splitData(df)
    assert y_train.shape[0] == X_train.shape[0], "Training sizes mismatched"
    assert y_test.shape[0] == X_test.shape[0], "Testing sizes mismatched"
    assert round(X_test.shape[0] * 0.8) == (y_test.shape[0]), "Test is not 20% of train"


def test_linearModelSK():
    lr = linearModelSK(df[["age", "children"]], df["charges"])
    assert abs(lr.coef_[0] - 50) < 0.001, "sklearn model coefficients do not match expected values"
    assert abs(lr.coef_[1] - 50) < 0.001, "sklearn model coefficients do not match expected values"


def test_linearModelSM():
    lr, preds = linearModelSM(df[["age", "children"]], df["charges"], df[["age", "children"]])
    assert (
        abs(lr.params.iloc[1] - 50) < 0.001
    ), "SM model coefficients do not match expected values"
    assert (
        abs(lr.params.iloc[2] - 50) < 0.001
    ), "SM model coefficients do not match expected values"
    assert preds.sum(), "Predictions are not numeric"


def test_pca():
    assert pca(df), "Error while scaling and fitting PCA"
