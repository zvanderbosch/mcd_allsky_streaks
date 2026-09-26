"""All-sky image processing routines

This script will process and analyze all-sky fisheye images
to detect streaks from satellites and potentially from other 
artificial and natural sources (e.g. planes, meteors, NEOs).

Author:
    Z. Vanderbosch (HET)

Last Updated:
    2026 Sept 25

"""

import os
import subprocess
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


@dataclass(frozen=True)
class ImageData:
    """Class to store all-sky image data, header, and path"""

    data: np.ndarray
    header: Mapping[str, Any] | None
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
        header: Mapping[str, Any] | None = None,
        save_to: str | Path | None = None
    ) -> ImageData:
        """
        Create new ImageData object from provided data + header,
        optionally save to a FITS file.
        """
        
        # Save to file if a path is given
        if save_to:
            if header:
                if isinstance(header,dict):
                    header = fits.Header(header)
                newHDU = fits.PrimaryHDU(data=data, header=header)
            else:
                newHDU = fits.PrimaryHDU(data=data)
            newHDU.writeto(save_to, overwrite=True)
            print(f'Saved to FITS file: {save_to}')
        
        return ImageData(
            data=data,
            header=header,
            path=save_to
        )
    


class PlateSolve:
    """Utilities for plate solving all-sky images"""

    @staticmethod
    def findStars(image: ImageData) -> str | Path:
        """
        Function to detect sources in an image and save XY positions
        to an output file in same directory as the image

        Parameters:
        -----------
        image: ImageData object
            Object containing image data, header, and source file name

        Returns:
        --------
        sourceFileXY: str or Path
            FITS table filename where XY positions are saved
        """

        # Get some image stats
        _,_,std = sigma_clipped_stats(
            image.data, sigma=3.0
        )

        # Find stars in image
        daoFinder = DAOStarFinder(
            threshold=4.0*std, 
            fwhm=3.0,
            peak_max=50000,
            n_brightest=100
        )
        daoSources = daoFinder(image.data)

        # Save sources to file
        sourceFileXY = f"{image.path.split('.')[0]}_xy.fits"
        sourceTable = Table(
            {
                'x':daoSources['x_centroid'],
                'y':daoSources['y_centroid']
            }
        )
        sourceTable.write(sourceFileXY, format='fits', overwrite=True)

        return sourceFileXY

    @staticmethod
    def solveWithANetOnline(
        xyFile: str | Path,
        xsize: int,
        ysize: int
    ):
        """
        Function that sends a list of XY pixel coordinates for
        detected objects, along with the image X and Y extentz,
        to the Astrometry.net online service to obtain the image
        plate solution as WCS header keyword values.

        Parameters:
        -----------
        xyFile: str
            FITS table filename containing source XY positions
        xsize: int
            Image width (pixels)
        ysize: int
            Image height (pixels)

        Returns:
        --------
        response.returncode: int
            Return code from subprocess
        """
        
        astDir = "/".join(xyFile.split("/")[0:-1])
        fileBase = xyFile.split("/")[-1].split(".")[0]

        # Plate solve using Astrometry.net client (anet_client.py)
        cmd = [
            'python', '/home/zach/HET/mcd_allsky_streaks/src/anet_client.py', 
            '--apikey', 'kdvqtjbqkbkbuyzb',
            '--upload-xy', f'{xyFile}',
            '--parity', '2',
            '--image-width', f'{xsize}',
            '--image-height', f'{ysize}',
            '--scale-units', 'arcsecperpix',
            '--scale-est', '70.0',
            '--scale-err', '10.0', # percent
            '--calibrate', f'{astDir}/{fileBase}_calib.txt',
            '--wcs', f'{astDir}/{fileBase}_wcs.fits',
            '--solve-time', '30.0'
        ]
        response = subprocess.run(cmd, timeout=None)
        if response.check_returncode():
            print(f'ERROR: Return code {response.returncode} from Astrometry.net API')
            
        return response.returncode

