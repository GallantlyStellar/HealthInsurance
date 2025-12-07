#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for health insurance portfolio project.

Test EDA functions. Paths assume tests
are run relative to the "src" directory.

GallantlyStellar
"""

import pandas as pd

from eda import ageChargeScatterplots


def test_ageChargeScatterplots():
    df = pd.DataFrame(
        {
            "age": [1, 2, 3, 4, 5],
            "charges": [100, 200, 300, 400, 500],
            "smoker": [0, 0, 1, 1, 0],
            "bmi": [25, 20, 35, 30, 32],
        }
    )
    assert ageChargeScatterplots(df), "Age-Charge Scatterplots failed to generate."
