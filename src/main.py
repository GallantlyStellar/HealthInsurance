#!/usr/bin/env python3
# coding: utf-8

"""
Assignment for Portfolio.

Regression analysis of health insurance information.

GallantlyStellar
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from importAndClean import importDF
from visualize import bivariate, univariate

# Make plot fonts bigger
plt.rcParams.update({"font.size": 20})

# read in data
df = importDF(path="../assets/data/raw/insurance.csv")
univariate(df, 3, 3)
bivariate(df, "charges", 2, 3)
# plt.show()
plt.close("all")
