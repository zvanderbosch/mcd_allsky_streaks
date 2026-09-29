"""Script used to test-run the image processing routines"""

import sys
sys.path.insert(0, "./src")

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from glob import glob
from functools import partial
from multiprocessing import Pool
from astropy.visualization import PercentileInterval
piv = PercentileInterval(90.)

# Local imports
import filepaths as fp
import printcolors as pc
from process_image import ImageIO
from process_image import fullPlateSolveANet

# Fall back to XCB to avoid QT5 wayland warnings
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Load the raw FITS images
fitsFile1 = f"{fp.dataDir}/ALPACA.2026-08-08T23:22:17.570.fits"
fitsFile2 = f"{fp.dataDir}/ALPACA.2026-08-08T23:24:21.850.fits"
image1 = ImageIO.load(fitsFile1)
image2 = ImageIO.load(fitsFile2)


# Divide the original images into a grid of smaller images
cs = 512  # crop size for stamps in pixel units
numrings = 5
ydim,xdim = image1.data.shape
xcen = np.floor(xdim/2)+1
ycen = np.floor(ydim/2)+1

rawStamps = []
solvedStamps = []
for ring in range(numrings):
    for xshift in range(-ring,ring+1):
        for yshift in range(-ring,ring+1):

            # Make sure the shift belongs to the given ring
            if (abs(xshift) < ring) and (abs(yshift) < ring):
                continue
            
            # Check whether stamp already exists
            stampPath = f"{fp.dataDir}/stamps/stamp_r{ring:02d}_{xshift:+03d}{yshift:+03d}.fits"
            stampPath = stampPath.replace("-","m").replace("+","p")
            if os.path.isfile(stampPath):
                stampImage = ImageIO.load(stampPath)
                if 'PLTSOLVD' in stampImage.header.keys():
                    solvedStamps.append(stampImage)
                else:
                    rawStamps.append(stampImage)
            else:
                # Create new ImageData object for stamp
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
partialSolver = partial(
    fullPlateSolveANet, 
    useExistingSolve=True,
    plotImage=True
)
with Pool(processes=fp.numThreads) as pool:
    results = pool.imap_unordered(partialSolver,rawStamps)
    for res in results:
        solvedStamps.append(res)

# Re-sort the stamps list 
stampFiles = [s.path for s in solvedStamps]
sortIndex = sorted(range(len(stampFiles)), key=lambda i: stampFiles[i])
solvedStamps = [solvedStamps[i] for i in sortIndex]


# Generate diagnostic figure for full-frame image
fig,ax = plt.subplots(figsize=(10,10))
vmin,vmax = piv.get_limits(image2.data[image2.data > 600])
ax.imshow(image2.data,vmin=vmin,vmax=vmax,cmap='Greys_r',origin='lower')
ringColors = ['r','royalblue','forestgreen','goldenrod','mediumpurple']

# Add rectangle patch to image
ringsLabeled = []
for stamp in solvedStamps:

    # Recreate ring/shift data from filename
    imgName = stamp.path.split('/')[-1]
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
    stampXcen = xcen + xshift*cs
    stampYcen = ycen + yshift*cs

    ringLabel = f'Ring {ringnum}'
    if ringLabel in ringsLabeled:
        ringLabel = '_none'
    else:
        ringsLabeled.append(ringLabel)
    
    # Add patch showing extent of image stamp
    rect = patches.Rectangle(
        (stampXcen - cs/2, stampYcen - cs/2), 
        cs, cs, 
        linewidth=2, 
        edgecolor=mcolors.to_hex(ringColors[ringnum])+'FF', 
        facecolor=mcolors.to_hex(ringColors[ringnum])+'22',
        zorder=10-ringnum,
        label=ringLabel
    )
    ax.add_patch(rect)

    # Add symbol indicating whether plate was solved or not
    if stamp.header['PLTSOLVD']:
        ax.scatter(stampXcen, stampYcen, marker='o', c='g', s=100)
    else:
        ax.scatter(stampXcen, stampYcen, marker='X', c='r', s=100)

ax.legend()
plt.savefig(f"{fp.dataDir}/full_frame_pltsolvd.png",dpi=100,bbox_inches='tight')
plt.close()

# Take image difference and create new ImageData object
# diffimPath = f"{DIRS['data']}/diffim.fits"
# diffimage = ImageIO.create(
#     data=image1.data-image2.data,
#     header=image2.header,
#     save_to=diffimPath
# )