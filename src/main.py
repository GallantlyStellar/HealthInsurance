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

# Make plot fonts bigger
plt.rcParams.update({"font.size": 20})

# read in data
df = importDF(path="../assets/data/raw/insurance.csv")
