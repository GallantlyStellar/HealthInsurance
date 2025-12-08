#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for health insurance portfolio project.

Test visualizations. Paths assume tests
are run relative to the "src" directory.

GallantlyStellar
"""

import pandas as pd

from visualize import bivariate, corr, univariate

df = pd.DataFrame(
    {
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "b": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
        "c": [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        "d": [25, 20, 35, 30, 32, 20, 22, 23, 40, 35, 30],
    }
)


def test_univariate():
    assert univariate(df), "Univariate plots failed to generate."


def test_bivariate():
    assert bivariate(df, "b"), "Bivariate plots failed to generate with continuous target."
    assert bivariate(df, "c"), "Bivariate plots failed to generate with discrete target."


def test_corr():
    assert corr(df), "Correlation heatmap failed to generate."
