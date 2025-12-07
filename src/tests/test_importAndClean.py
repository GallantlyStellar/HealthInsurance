#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for health insurance portfolio project.

Test data import and cleaning process. Paths assume tests
are run relative to the "src" directory.

GallantlyStellar
"""

from hashlib import file_digest

from importAndClean import importDF


def test_filehash():
    """Ensure the source data has not changed."""
    with open("../assets/data/raw/insurance.csv", "rb") as file:
        digest = file_digest(file, "sha256").hexdigest()
    # Compare to known filehash with sha256sum of original file
    gnuShaSumsHash = "388eff679557d08ac19f463d025de5e0b4adc482537c8456d19934d78621fd47"
    assert digest == gnuShaSumsHash, "Check source data for alterations, hash mismatch"


def test_importDF():
    """Test properties of imported dataframe."""
    df = importDF(path="../assets/data/raw/insurance.csv")
    colNames = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    assert all(df.columns == colNames), "Column labels have changed"
    assert all(
        df.dtypes
        == [
            "uint8",
            "uint8",
            "float64",
            "uint8",
            "uint8",
            "category",
            "float64",
        ]
    ), "Dtypes have changed"
    assert df.shape == (1338, 7), "DF shape has changed"
