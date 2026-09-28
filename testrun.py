"""Script used to test-run the image processing routines"""

import sys
sys.path.insert(0, "./src")

import numpy as np

from glob import glob
from functools import partial
from multiprocessing import Pool

# Local imports
import filepaths as fp
import printcolors as pc
from process_image import ImageIO
from process_image import fullPlateSolveANet


# Load the raw FITS images
fitsFile1 = f"{fp.dataDir}/ALPACA.2026-08-08T23:22:17.570.fits"
fitsFile2 = f"{fp.dataDir}/ALPACA.2026-08-08T23:24:21.850.fits"
image1 = ImageIO.load(fitsFile1)
image2 = ImageIO.load(fitsFile2)


# Divide the original images into a grid of smaller images
cs = 512  # crop size for stamps in pixel units
numrings = 3
ydim,xdim = image1.data.shape
xcen = np.floor(xdim/2)+1
ycen = np.floor(ydim/2)+1

rawStamps = []
for ring in range(numrings):
    for xshift in range(-ring,ring+1):
        for yshift in range(-ring,ring+1):

            # Make sure the shift belongs to the given ring
            if (abs(xshift) < ring) and (abs(yshift) < ring):
                continue

            # Create new ImageData object for stamp
            stampPath = f"{fp.dataDir}/stamps/stamp_r{ring:02d}_{xshift:+03d}{yshift:+03d}.fits"
            stampPath = stampPath.replace("-","m").replace("+","p")
            stampXcen = xcen + xshift*cs
            stampYcen = ycen + yshift*cs
            stampImage = ImageIO.create(
                data=image2.data[
                    int(stampYcen-cs/2):int(stampYcen+cs/2),
                    int(stampXcen-cs/2):int(stampXcen+cs/2)
                ],
                save_to=stampPath
            )
            rawStamps.append(stampImage)


# Solve all images, using multiprocessing
solvedStamps = []
partialSolver = partial(
    fullPlateSolveANet, 
    useExistingSolve=True,
    plotImage=True
)
with Pool(processes=fp.numThreads) as pool:
    results = pool.imap_unordered(partialSolver,rawStamps)
    for res in results:
        solvedStamps.append(res)



# Take image difference and create new ImageData object
# diffimPath = f"{DIRS['data']}/diffim.fits"
# diffimage = ImageIO.create(
#     data=image1.data-image2.data,
#     header=image2.header,
#     save_to=diffimPath
# )