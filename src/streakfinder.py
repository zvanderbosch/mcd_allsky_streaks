"""
Streak detection routines

This script contains routines needed for detecting
streaks from both artificial and natural sources.

Author:
    Z. Vanderbosch (HET)

Last Updated:
    2026 Sept 29
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import canny
from skimage.transform import hough_line
from skimage.transform import hough_line_peaks
from astropy.visualization import PercentileInterval

# Local imports
import printcolors as pc
from imageproc import ImageData
from pyradon.finder import Finder


# Define print status prefix
scriptName = 'streakfinder.py'
PREFIX = f'{pc.GREEN}{scriptName:19s}{pc.END}: '


class streakFinding:
    """
    Utilities for streak detection
    """

    @staticmethod
    def houghStreaks(
            image: ImageData,
            edgeSigma: float = 3.0,
            edgeLowThreshold: float | None = 10.,
            edgeHighThreshold: float | None = 20.,
            peakThreshold: float | None = None,
            plotStreaks: bool = True
    ) -> dict:
        """
        Streak detection using Hough transform

        Parameters:
        -----------
        image: ImageData
            Image to perform streak detection on
        edgeSigma: float
            StDev of gaussian used for image smoothing before edge detection
        edgeLowThreshold: float
            Lower bound for hysteresis thresholding (edge linking)
        edgeHighThreshold: float
            Upper bound for hysteresis thresholding (edge linking)

        Returns:
        --------
        houghResults: dict
            Parameters of the detected streaks, including:
                hough_accum: Pixel sum along the line
                hough_theta: Line angle
                hough_dist : Line x-intercept
        """

        # Perform edge detections
        edges = canny(
            image.data, 
            sigma=edgeSigma, 
            low_threshold=edgeLowThreshold, 
            high_threshold=edgeHighThreshold
        )

        # Generate array of test angle from -90 to +90 deg with 0.5-deg steps
        testAngles = np.linspace(-np.pi/2, np.pi/2, 360, endpoint=False)

        # Classic straight-line Hough transform
        houghSpace, theta, distance = hough_line(
            edges, theta=testAngles
        )

        # Detect peaks in the Hough transform sinogram
        if peakThreshold:
            sum, angles, dists = hough_line_peaks(
                houghSpace, 
                theta, 
                distance,
                threshold=peakThreshold
            )
        else:
            sum, angles, dists = hough_line_peaks(
                houghSpace, 
                theta, 
                distance
            )

        # Package results into dict
        houghResults = {
            'hough_accum': sum,
            'hough_theta': angles,
            'hough_dist': dists
        }

        # Plot streaks over image
        if plotStreaks:

            # Define image scaling object
            piv = PercentileInterval(95.)

            # Creating plotting axis
            _,ax = plt.subplots(
                figsize=(10,10)
            )

            # Plot image
            vmin,vmax = piv.get_limits(image.data)
            ax.imshow(
                image.data, 
                vmin=vmin, 
                vmax=vmax, 
                cmap='Greys_r', 
                origin='lower'
            )

            for accum, angle, rdist in zip(sum, angles, dists):
                if accum > 50:
                    (x0, y0) = rdist * np.array([np.cos(angle), np.sin(angle)])
                    ax.axline(
                        (x0, y0), 
                        slope=np.tan(angle + np.pi / 2), 
                        c='b', 
                        lw=2, 
                        alpha=0.75
                    )

            ax.set_xlim(0,image.data.shape[1])
            ax.set_ylim(0,image.data.shape[0])

            plt.show()
            plt.close()


        return houghResults
    
    @staticmethod
    def radonStreaks(
            image: ImageData,
            threshold: float,
            psf: float | np.ndarray,
            variance: float | np.ndarray,
            meanSubtract: bool = False,
            verbosity: int = 1,
            plotStreaks: bool = False
    ):
        """
        Streak detection using Radon transform as described
        by Nir et al. 2018 and implemented in python here:
        https://github.com/guynir42/pyradon

        Nir et al. 2018: https://ui.adsabs.harvard.edu/abs/2018AJ....156..229N

        Parameters:
        -----------
        image: ImageData
            Image to perform streak detection on
        threshold: float
            S/N threshold for detected streaks
        psf: float | array
            Scalar PSF FWHM or small array representing the PSF
        variance: float | array
            Scalar representing mean image variance or an array
            providing the variance map for the input image.
        meanSubtract: bool
            Whether to subtract image mean before streak detection.
            (Default = False)
        verbosity: int
            Print verbosity of the streak finder. 1 = min, >1 = max.
            (Default = 1)

        Returns:
        --------
        
        """

        # Define print status prefix
        funcName = 'streakFinding.radonStreaks'
        funcPrefix = f'{PREFIX}({pc.ORANGE}{funcName}{pc.END}) '

        # Setup the pyradon streak finder
        pyradonFinder = Finder(
            verbosity=verbosity, 
            threshold=threshold,
            use_exclude=False,
            use_subtract_mean=meanSubtract
        )

        # Provide input data to finder
        pyradonFinder.input(
            image.data,
            variance=variance,
            psf=psf
        )

        Nstreaks = len(pyradonFinder.streaks)
        print(f"{funcPrefix}Found {Nstreaks} streaks in image {image.path.split('/')[-1]}")
        for i in range(Nstreaks):
            pyradonFinder.streaks[i].print()


        # Plot streaks over image
        if plotStreaks and (Nstreaks > 0):


            # Define image scaling object
            piv = PercentileInterval(95.)

            lineFormats = [
                '--m',
                '--c',
                '--r',
                '--b',
                '--g'
            ]

            # Creating plotting axis
            _,ax = plt.subplots(
                figsize=(10,10)
            )

            # Plot image
            vmin,vmax = piv.get_limits(image.data)
            ax.imshow(
                image.data, 
                vmin=vmin, 
                vmax=vmax, 
                cmap='Greys_r', 
                origin='lower'
            )

            for i in range(Nstreaks):
                pyradonFinder.streaks[i].plot_lines(
                    ax=ax,
                    offset=5,
                    line_format=lineFormats[i%5]
                )

            ax.set_xlim(0,image.data.shape[1])
            ax.set_ylim(0,image.data.shape[0])

            plt.show()
            plt.close()

        return
