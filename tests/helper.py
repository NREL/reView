# -*- coding: utf-8 -*-
"""Helper functions for tests"""
import PIL
import imagehash
import numpy as np


def compare_images_approx(
    image_1_path, image_2_path, hash_size=12, max_diff_pct=0.25
):
    """
    Check if two images match approximately.

    Parameters
    ----------
    image_1_path : pathlib.Path
        File path to first image.
    image_2_path : pathlib.Path
        File path to first image.
    hash_size : int, optional
        Size of the image hashes that will be used for image comparison,
        by default 12. Increase to make the check more precise, decrease to
        make it more approximate.
    max_diff_pct : float, optional
        Tolerance for the amount of difference allowed, by default 0.05 (= 5%).
        Increase to allow for a larger delta between the image hashes, decrease
        to make the check stricter and require a smaller delta between the
        image hashes.

    Returns
    -------
    bool
        Returns true if the images match approximately, false if not.
    """

    expected_hash = imagehash.phash(
        PIL.Image.open(image_1_path), hash_size=hash_size
    )
    out_hash = imagehash.phash(
        PIL.Image.open(image_2_path), hash_size=hash_size
    )

    max_diff_bits = int(np.ceil(hash_size * max_diff_pct))

    diff = expected_hash - out_hash
    matches = diff <= max_diff_bits
    pct_diff = float(diff) / hash_size

    return matches, pct_diff
