#!/usr/bin/env python3
# coding: utf-8

"""
Supports analysis for health insurance portfolio project.

Test data import and cleaning process. Paths assume tests
are run relative to the "src" directory.

GallantlyStellar
"""

from hashlib import file_digest


def test_filehash():
    """Ensure the source data has not changed."""
    with open("../assets/data/raw/", "rb") as file:
        digest = file_digest(file, "sha256").hexdigest()
    # Compare to known filehash with sha256sum of original file
    gnuShaSumsHash = "714b6515723942583cb876b989275c13e13b26409159d0074a281c53767c58f5"
    assert digest == gnuShaSumsHash, "Check source data for alterations, hash mismatch"
