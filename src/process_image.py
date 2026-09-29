"""
All-sky image processing routines

This script will process and analyze all-sky fisheye images
to detect streaks from satellites and potentially from other 
artificial and natural sources (e.g. planes, meteors, NEOs).

Author:
    Z. Vanderbosch (HET)

Last Updated:
    2026 Sept 29

"""

import os
import json
import subprocess
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.visualization import PercentileInterval
from photutils.detection import DAOStarFinder

# Local imports
import filepaths as fp
import printcolors as pc

# Fall back to XCB to avoid QT5 wayland warnings
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Define print status prefix
scriptName = 'process_image.py'
PREFIX = f'{pc.GREEN}{scriptName:19s}{pc.END}: '


@dataclass(frozen=True)
class ImageData:
    """Class to store all-sky image data, header, and path"""

    data: np.ndarray
    header: Mapping[str, Any] | None
    path: str | None


class ImageIO:
    """Utilities for various FITS image IO tasks"""

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
            header = hdul[0].header
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

        # Define print status prefix
        funcName = 'ImageIO.create'
        funcPrefix = f'{PREFIX}({pc.ORANGE}{funcName}{pc.END}) '
        
        # Save to file if a path is given
        if save_to:
            if header:
                if isinstance(header,dict):
                    header = fits.Header(header)
                newHDU = fits.PrimaryHDU(data=data, header=header)
            else:
                newHDU = fits.PrimaryHDU(data=data)
            newHDU.writeto(save_to, overwrite=True)
            print(f"{funcPrefix}Saved image to FITS file: {save_to.split('/')[-1]}")
        
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
        ysize: int,
        useExisting: bool = False
    ) -> str | None:
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
        useExisting: bool
            Use existing astrometry.net solutions (default = False)

        Returns:
        --------
        wcsFile: str | None
            Path to downloaded wcs header file, or None if solving failed
        """

        # Define print status prefix
        funcName = 'PlateSolve.solveWithANetOnline'
        funcPrefix = f'{PREFIX}({pc.ORANGE}{funcName}{pc.END}) '
        
        # Get the working directory and base filename
        xyFile = str(xyFile)
        astDir = "/".join(xyFile.split("/")[0:-1])
        fileBase = xyFile.split("/")[-1].split(".")[0]

        # Define paths to output files
        wcsFile = f'{astDir}/{fileBase}_wcs.fits'
        calibFile = f'{astDir}/{fileBase}_calib.txt'

        # Use existing solution if desired
        if useExisting:
            if os.path.isfile(wcsFile) and os.path.isfile(calibFile):
                return wcsFile, calibFile

        # Plate solve using Astrometry.net client (anet_client.py)
        cmd = [
            'python', '/home/zach/HET/mcd_allsky_streaks/src/anet_client.py', 
            '--apikey', fp.anetAPIKey,
            '--upload-xy', xyFile,
            '--parity', '2',
            '--image-width', f'{xsize}',
            '--image-height', f'{ysize}',
            '--scale-units', 'arcsecperpix',
            '--scale-est', '70.0',
            '--scale-err', '10.0', # percent
            '--calibrate', calibFile,
            '--wcs', wcsFile,
            '--solve-time', '120.0',
            '--crpix-center'
        ]
        try:
            print(f'{funcPrefix}Sending Astrometry.net command for {xyFile.split("/")[-1]}')
            response = subprocess.run(cmd, timeout=None)
            response.check_returncode()
        except:
            print(f'{funcPrefix}{pc.RED}ERROR{pc.END}: Return code {response.returncode} from Astrometry.net API')
            return None, None
        
        return wcsFile, calibFile

    @staticmethod
    def updateFITSHeader(
            imageFile: str | Path,
            wcsFile: str | Path | None,
            calibFile: str | Path | None
    ) -> str | Path:
        """
        Function that updates a FITS image file's header with
        the WCS info contained in the provided WCS header file.

        Parameters:
        -----------
        imageFile: str | Path
            Path to FITS image file
        wcsFile: str | Path | None
            Path to FITS WCS file. If None, assumes plate solving failed.
        calibFile: str | Path | None
            Path to JSON calibration file. If None, assumes plate solving failed.

        Returns:
        --------
        imageFile: str | Path
            Return path to updated FITS image file, same as the input file.
        """

        # WCS Solution FITS header keys to save into image headers
        wcsKeys = [
            'WCSAXES', 'CTYPE1', 'CTYPE2','EQUINOX','LONPOLE','LATPOLE',
            'CRVAL1','CRVAL2','CRPIX1','CRPIX2','CUNIT1','CUNIT2',
            'CD1_1','CD1_2','CD2_1','CD2_2',
            'A_ORDER','A_0_0','A_0_1','A_0_2','A_1_0','A_1_1','A_2_0',
            'B_ORDER','B_0_0','B_0_1','B_0_2','B_1_0','B_1_1','B_2_0',
            'AP_ORDER','AP_0_0','AP_0_1','AP_0_2','AP_1_0','AP_1_1','AP_2_0',
            'BP_ORDER','BP_0_0','BP_0_1','BP_0_2','BP_1_0','BP_1_1','BP_2_0'
        ]

        # If wcsFile is not present, add keyword to indicate solution failed
        if (wcsFile is None) or (calibFile is None):
            with fits.open(imageFile, mode='update') as hdul:
                hdul[0].header['PLTSOLVD'] = (False, 'Astrometric solution solved')
                hdul.flush()
            return imageFile
        
        # Load the WCS FITS headers
        with fits.open(wcsFile) as hdul:
            wcsHDR = hdul[0].header

        # Create coordinate object using CRVAL values
        imgCoord = SkyCoord(
            ra=wcsHDR['CRVAL1']*u.deg,
            dec=wcsHDR['CRVAL2']*u.deg,
            frame='icrs'
        )

        # Get pixel scale from the calibration file
        with open(calibFile) as js:
            calib = json.load(js)
        pixscale = calib['pixscale'] # [arcsec/pix] platescale

        # Update the image FITS header
        with fits.open(imageFile, uint=False, mode='update') as hdul:
            imgHDR = hdul[0].header
            imgHDR['PLTSOLVD'] = (True, 'Astrometric solution solved')
            for key in wcsKeys:
                if key not in list(wcsHDR.keys()):
                    continue
                imgHDR[key] = (wcsHDR[key], wcsHDR.comments[key])

            # Add RA and DEC values to the header
            imgHDR.set(
                'RA',
                imgCoord.ra.to_string(unit='hour',sep=':',precision=2),
                'Right Ascension at image center',
                before='WCSAXES'
            )
            imgHDR.set(
                'DEC',
                imgCoord.dec.to_string(unit='deg',sep=':',precision=2),
                'Declintation at image center',
                before='WCSAXES'
            )
            
            # Add pixel scale to header values
            imgHDR.set(
                'PIXSCALE', 
                pixscale, 
                '[deg/pixel] plate scale', 
                before='WCSAXES'
            )

            # Add history
            if 'HISTORY' not in imgHDR:
                imgHDR['HISTORY'] = 'WCS created using the Astrometry.net suite'
                imgHDR['HISTORY'] = wcsHDR['HISTORY'][1]
                imgHDR['HISTORY'] = wcsHDR['HISTORY'][2]

            # Flush changes to file
            hdul.flush()

        return imageFile


def fullPlateSolveANet(
        image: ImageData,
        useExistingSolve: bool = False,
        plotImage: bool = False
) -> ImageData:
    """
    A convenience function that executes multiple steps needed
    for plate solving an image using Astrometry.net online.

    1) Detect sources, output positions to FITS table file
    2) Submit source positions and image dimensions to Astrometry.net
       for plate solving, downloading the results when finished.
    3) Update the original image header with WCS header info.
    4) Load and return the updated image.

    Parameters:
    -----------
    image: ImageData
        Image to be plate solved.
    useExistingSolve: bool
        Use existing astrometry.net solutions (default = False)
    plotImage: bool
        WHether to save a plot of the solved image (default = False)

    Returns:
    --------
    imageSolved: ImageData
        Same image provided as input, but with header updated to
        include the results of the plate solution. If plate solving
        succeeded, header will contain many new WCS keywords. If 
        plate solving failed, only the header value PLTSOLVD = F
        will be added to indicate the plate solving failed.
    """

    # Get image name
    imgName = image.path.split('/')[-1]

    # Define print status prefix
    funcName = 'fullPlateSolveANet'
    funcPrefix = f'{PREFIX}({pc.ORANGE}{funcName}{pc.END}) '
    print(f"{funcPrefix}Starting plate solving for {imgName}")

    # Get image dimensions
    ny,nx = image.data.shape

    # Get source XY positions
    sourceFileXY = PlateSolve.findStars(image)

    # Plate solve using source XY positions
    wcsPath, calibPath = PlateSolve.solveWithANetOnline(
        sourceFileXY, nx, ny, useExisting=useExistingSolve
    )

    # Update the saved FITS file
    _ = PlateSolve.updateFITSHeader(
        image.path, wcsPath, calibPath
    )

    # Load in the updated FITS image
    imageSolved = ImageIO.load(image.path)

    # Plot stamp image
    if plotImage:
        print(f"{funcPrefix}Generating figure for {imgName}")

        # Generate filename for the saved figure
        figFile = f'{image.path.split(".")[0]}.png'

        # Load source positions used for solving
        with fits.open(sourceFileXY) as hdul:
            sourceTable = hdul[1].data

        # Get metadata from image file name
        ringnum = int(imgName.split("_")[1].strip("r"))
        xyshift = imgName.split("_")[2]
        if xyshift[0] == 'p':
            xshift = int(imgName.split("_")[2][1:3])
        else:
            xshift = -int(imgName.split("_")[2][1:3])
        if xyshift[3] == 'p':
            yshift = int(imgName.split("_")[2][4:6])
        else:
            yshift = -int(imgName.split("_")[2][4:6])

        # Define image scaling object
        piv = PercentileInterval(95.)

        # Generate WCS object is PLTSOLVD = True
        plateSolved = imageSolved.header['PLTSOLVD']
        if plateSolved:
            wcs = WCS(imageSolved.header)
            _,ax = plt.subplots(
                figsize=(10,10),
                subplot_kw=dict(projection=wcs)
            )
        else:
            _,ax = plt.subplots(
                figsize=(10,10)
            )
        
        # Plot the image
        vmin,vmax = piv.get_limits(imageSolved.data)
        ax.imshow(imageSolved.data,vmin=vmin,vmax=vmax,cmap='Greys_r')

        # Plot detected sources
        ax.scatter(
            sourceTable.x,
            sourceTable.y,
            marker='o', fc='None', ec='c', s=125, lw=2
        )

        # Add grid for plate solved images only
        if plateSolved:
            ax.grid(ls='-',color='indianred',lw=1.5)

        # Add title
        pltTitle = (
            f"{image.path.split('/')[-1]} "
            f"(PLTSOLVD = {plateSolved})\n"
            f"{nx}x{ny} Stamp for Ring {ringnum}, "
            f"X-shift {xshift:+d}, "
            f"Y-shift {yshift:+d}"
        )
        ax.set_title(pltTitle,fontsize=18)

        # Change axis labels
        if 'ra' in ax.get_xlabel():
            ax.set_xlabel('Right Ascension', fontsize=14)
            ax.set_ylabel('Declination', fontsize=14)
        elif 'ra' in ax.get_ylabel():
            ax.set_xlabel('Declination', fontsize=14)
            ax.set_ylabel('Right Ascension', fontsize=14)
        else:
            ax.set_xlabel('X (pixels)', fontsize=14)
            ax.set_ylabel('Y (pixels)', fontsize=14)

        # Save figure
        plt.savefig(figFile, dpi=200, bbox_inches='tight')
        plt.close()

    return imageSolved