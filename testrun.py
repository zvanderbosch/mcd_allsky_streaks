"""
Script used to test-run the image processing routines

Author:
    Z. Vanderbosch (HET)

Last Updated:
    2026 Sept 29
"""

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
from astropy.stats import sigma_clipped_stats
from astropy.visualization import PercentileInterval
piv = PercentileInterval(90.)

# Local imports
import filepaths as fp
import printcolors as pc
from imageproc import ImageIO
from imageproc import fullPlateSolveANet
from streakfinder import streakFinding

# Fall back to XCB to avoid QT5 wayland warnings
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Load the raw FITS images
fitsFile1 = f"{fp.dataDir}/ALPACA.2026-08-08T23:22:17.570.fits"
fitsFile2 = f"{fp.dataDir}/ALPACA.2026-08-08T23:24:21.850.fits"
image1 = ImageIO.load(fitsFile1)
image2 = ImageIO.load(fitsFile2)


# Divide the original images into a grid of smaller images
cs = 512  # crop size for stamps in pixel units
numrings = 6
ydim,xdim = image1.data.shape
xcen = np.floor(xdim/2)+1
ycen = np.floor(ydim/2)+1

rawStamps1 = []
rawStamps2 = []
diffStamps = []
solvedStamps = []
for ring in range(numrings):
    for xshift in range(-ring,ring+1):
        for yshift in range(-ring,ring+1):

            # Make sure the shift belongs to the given ring
            if (abs(xshift) < ring) and (abs(yshift) < ring):
                continue

            # Get file names for each stamp
            stampPath1 = f"{fp.dataDir}/stamps/stamp_r{ring:02d}_{xshift:+03d}{yshift:+03d}_1.fits"
            stampPath2 = f"{fp.dataDir}/stamps/stamp_r{ring:02d}_{xshift:+03d}{yshift:+03d}_2.fits"
            stampPath1 = stampPath1.replace("-","m").replace("+","p")
            stampPath2 = stampPath2.replace("-","m").replace("+","p")

            # Create new ImageData object for stamp
            stampXcen = xcen + xshift*cs
            stampYcen = ycen + yshift*cs
            stampImage1 = ImageIO.create(
                data=image1.data[
                    int(stampYcen-cs/2):int(stampYcen+cs/2),
                    int(stampXcen-cs/2):int(stampXcen+cs/2)
                ],
                #save_to=stampPath1
            )
            stampImage2 = ImageIO.create(
                data=image2.data[
                    int(stampYcen-cs/2):int(stampYcen+cs/2),
                    int(stampXcen-cs/2):int(stampXcen+cs/2)
                ],
                #save_to=stampPath2
            )
            rawStamps1.append(stampImage1)
            rawStamps2.append(stampImage2)

            # Take image difference and create new ImageData object
            diffimPath = f"{fp.dataDir}/stamps/stamp_r{ring:02d}_{xshift:+03d}{yshift:+03d}_diff.fits"
            diffimPath = diffimPath.replace("-","m").replace("+","p")
            diffImage = ImageIO.create(
                data=stampImage2.data-stampImage1.data,
                header=image2.header,
                save_to=diffimPath
            )
            diffStamps.append(diffImage)
            
            # # Check whether stamp already exists
            # if os.path.isfile(stampPath2):
            #     stampImage2 = ImageIO.load(stampPath2)
            #     if 'PLTSOLVD' in stampImage2.header.keys():
            #         solvedStamps.append(stampImage2)
            #     else:
            #         rawStamps.append(stampImage2)
            # else:
            #     # Create new ImageData object for stamp
            #     stampXcen = xcen + xshift*cs
            #     stampYcen = ycen + yshift*cs
            #     stampImage2 = ImageIO.create(
            #         data=image2.data[
            #             int(stampYcen-cs/2):int(stampYcen+cs/2),
            #             int(stampXcen-cs/2):int(stampXcen+cs/2)
            #         ],
            #         save_to=stampPath2
            #     )
            #     rawStamps.append(stampImage2)


# Perform streak detection on difference images
# for image in diffStamps:

    # houghStreaks = streakFinding.houghStreaks(
    #     image,
    #     edgeSigma=3.5,
    #     edgeLowThreshold=10.,
    #     edgeHighThreshold=20.,
    #     peakThreshold=30.
    # )
    # _,_,imgStd = sigma_clipped_stats(
    #     image.data, sigma=5.0
    # )
    # streakFinding.radonStreaks(
    #     image,
    #     threshold=10.,
    #     psf=3.0,
    #     variance=imgStd**2,
    #     meanSubtract=True,
    #     plotStreaks=True
    # )


# # Solve all images, using multiprocessing
# partialSolver = partial(
#     fullPlateSolveANet, 
#     useExistingSolve=True,
#     plotImage=True
# )
# with Pool(processes=fp.numThreads) as pool:
#     results = pool.imap_unordered(partialSolver,rawStamps)
#     for res in results:
#         solvedStamps.append(res)

# Re-sort the stamps list 
# stampFiles = [s.path for s in solvedStamps]
# sortIndex = sorted(range(len(stampFiles)), key=lambda i: stampFiles[i])
# solvedStamps = [solvedStamps[i] for i in sortIndex]


# Search for streaks using Hough transform



# # Generate diagnostic figure for full-frame image
# fig,ax = plt.subplots(figsize=(10,10))
# vmin,vmax = piv.get_limits(image2.data[image2.data > 600])
# ax.imshow(image2.data,vmin=vmin,vmax=vmax,cmap='Greys_r',origin='lower')
# ringColors = ['r','royalblue','forestgreen','goldenrod','mediumpurple']

# # Add rectangle patch to image
# ringsLabeled = []
# for stamp in solvedStamps:

#     # Recreate ring/shift data from filename
#     imgName = stamp.path.split('/')[-1]
#     ringnum = int(imgName.split("_")[1].strip("r"))
#     xyshift = imgName.split("_")[2]
#     if xyshift[0] == 'p':
#         xshift = int(imgName.split("_")[2][1:3])
#     else:
#         xshift = -int(imgName.split("_")[2][1:3])
#     if xyshift[3] == 'p':
#         yshift = int(imgName.split("_")[2][4:6])
#     else:
#         yshift = -int(imgName.split("_")[2][4:6])
#     stampXcen = xcen + xshift*cs
#     stampYcen = ycen + yshift*cs

#     ringLabel = f'Ring {ringnum}'
#     if ringLabel in ringsLabeled:
#         ringLabel = '_none'
#     else:
#         ringsLabeled.append(ringLabel)
    
#     # Add patch showing extent of image stamp
#     rect = patches.Rectangle(
#         (stampXcen - cs/2, stampYcen - cs/2), 
#         cs, cs, 
#         linewidth=2, 
#         edgecolor=mcolors.to_hex(ringColors[ringnum])+'FF', 
#         facecolor=mcolors.to_hex(ringColors[ringnum])+'22',
#         zorder=10-ringnum,
#         label=ringLabel
#     )
#     ax.add_patch(rect)

#     # Add symbol indicating whether plate was solved or not
#     if stamp.header['PLTSOLVD']:
#         ax.scatter(stampXcen, stampYcen, marker='o', c='g', s=100)
#     else:
#         ax.scatter(stampXcen, stampYcen, marker='X', c='r', s=100)

# ax.legend()
# plt.savefig(f"{fp.dataDir}/full_frame_pltsolvd.png",dpi=100,bbox_inches='tight')
# plt.close()
