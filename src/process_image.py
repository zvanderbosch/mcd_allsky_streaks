"""All-sky image processing routines

This script will process and analyze all-sky fisheye images
to detect streaks from satellites and potentially from other 
artificial and natural sources like planes, meteors, & NEOs.

Usage:
    TBD

Arguments:
    TBD

Options:
    TBD

Input Files Required:
    TBD

Author:
    Z. Vanderbosch (HET)

Last Updated:
    2026 Sept 24

"""

import os
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping
from astropy.io import fits


@dataclass(frozen=True)
class ImageData:
    """Class to store all-sky image data, header, and path"""

    data: np.ndarray
    header: Mapping[str, Any]
    path: str | None


class ImageIO:


    @staticmethod
    def load(source: str | Path) -> ImageData:
        """
        Load primary-HDU image data and header for one FITS file.
        """

        # Get path to FITS file
        fits_source = str(source)
        if not os.path.isfile(fits_source):
            raise FileExistsError(f"FITS file not found: {fits_source}")
        
        # Open FITS file and retrieve data/header
        with fits.open(fits_source, memmap=False) as hdul:
            header = dict(hdul[0].header)
            data = np.array(hdul[0].data, copy=True).astype(np.float64)

        # Check that data is not empty
        if data.ndim == 0:
            raise ValueError(f"Primary HDU has no pixel data: {fits_source}")
        
        return ImageData(
            data=data,
            header=header,
            path=fits_source
        )
    

    @staticmethod
    def create(
        data: np.ndarray,
        header: Mapping[str, Any],
        save_to: str | Path | None = None
    ) -> ImageData:
        """
        Create new ImageData object from provided data + header,
        optionally save to a FITS file.
        """
        
        # Save to file if a path is given
        if save_to:
            print(f'Saved to FITS file: {save_to}')
            # Need to add in actual save logic here
        
        return ImageData(
            data=data,
            header=header,
            path=save_to
        )