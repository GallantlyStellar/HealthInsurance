#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for health insurance portfolio project.

Fit linear model.

GallantlyStellar
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def oneHot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode labels, dropping one instance to prevent multicolinearity"""
    return pd.get_dummies(df, drop_first=True)


def splitData(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split encoded DF into train and testing sets."""
    return train_test_split(
        df.drop("charges", axis=1),
        df["charges"],
        test_size=0.2,
        random_state=7406,
    )


def linearModel(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit multiple linear regression model to the encoded, split training data."""
    return LinearRegression().fit(X_train, y_train)


def pca(df: pd.DataFrame) -> PCA:
    """Standardize the data and perform principal component analysis."""
    scaled = StandardScaler().fit_transform(df)
    return PCA().fit(scaled)
