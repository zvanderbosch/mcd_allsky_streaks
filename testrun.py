"""Script used to test-run the image processing routines"""

import sys

sys.path.insert(0, "./src")

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from glob import glob
from astropy.visualization import PercentileInterval

from process_image import ImageIO
from process_image import PlateSolve


# Define some base directories
BASE = '/home/zach/HET'
DIRS = {
    'base':f'{BASE}',
    'work':f'{BASE}/mcd_allsky_streaks',
    'data':f'{BASE}/mcd_allsky_streaks/data',
    'conf':f'{BASE}/mcd_allsky_streaks/config'
}


# Load the raw FITS images
fitsFile1 = f"{DIRS['data']}/ALPACA.2026-08-08T23:22:17.570.fits"
fitsFile2 = f"{DIRS['data']}/ALPACA.2026-08-08T23:24:21.850.fits"
image1 = ImageIO.load(fitsFile1)
image2 = ImageIO.load(fitsFile2)


# Divide the original images into a grid of smaller images
cs = 256  # crop size for stamps in pixel units
numrings = 1
ydim,xdim = image1.data.shape
xcen = np.floor(xdim/2)+1
ycen = np.floor(ydim/2)+1

for ring in range(numrings):
    for xshift in range(-ring,ring+1):
        for yshift in range(-ring,ring+1):

            # Make sure the shift belongs to the given ring
            if (abs(xshift) < ring) and (abs(yshift) < ring):
                continue

            # Create new ImageData object for stamp
            stampPath = f"{DIRS['data']}/stamps/stamp_r{ring:02d}_{xshift:+03d}{yshift:+03d}.fits"
            stampPath = stampPath.replace("-","m").replace("+","p")
            stampXcen = xcen + xshift*cs
            stampYcen = ycen + yshift*cs
            stampImage = ImageIO.create(
                data = image2.data[
                    int(stampYcen-cs/2):int(stampYcen+cs/2),
                    int(stampXcen-cs/2):int(stampXcen+cs/2)
                ],
                save_to=stampPath
            )

            # Get source XY positions
            sourceFileXY = PlateSolve.findStars(stampImage)

            # Plate solve using source XY positions
            responseCode = PlateSolve.solveWithANetOnline(sourceFileXY,cs,cs)


# Take image difference and create new ImageData object
# diffimPath = f"{DIRS['data']}/diffim.fits"
# diffimage = ImageIO.create(
#     data=image1.data-image2.data,
#     header=image2.header,
#     save_to=diffimPath
# )