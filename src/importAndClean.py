#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for health insurance portfolio project.

Modules to import and clean the insurance data.

GallantlyStellar
"""

import pandas as pd


def importDF(path: str = "../assets/data/raw/insurance.csv") -> pd.DataFrame:
    """Format a pandas dataframe.

    Parameters:
        path: string
            Location of the dataset to import

    Returns:
        df: pd.DataFrame
            Formatted dataframe for further analysis
    """
    df = pd.read_csv(
        "../assets/data/raw/insurance.csv",
        dtype={
            "age": "uint8",
            "sex": "category",
            "bmi": "float64",
            "children": "uint8",
            "smoker": "category",
            "region": "category",
            "charges": "float64",
        },
    )

    # Label encode genders in alignment with ISO 5218
    df["sex"] = (
        df["sex"]
        .cat.rename_categories(
            {
                "male": 0,
                "female": 1,
            }
        )
        .astype("uint8")
    )

    # Label encode smoker history; int not bool allows use in regression
    df["smoker"] = (
        df["smoker"]
        .cat.rename_categories(
            {
                "no": 0,
                "yes": 1,
            }
        )
        .astype("uint8")
    )

    # Indices 195 and 581 are identical; improbable these are different individuals
    df = df.drop_duplicates()

    return df
