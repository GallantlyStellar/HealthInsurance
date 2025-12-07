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

# read in data
df = pd.read_csv("../data/raw/data.csv", dtype={"age": "uint8"})

# Make plot fonts bigger
plt.rcParams.update({"font.size": 20})
