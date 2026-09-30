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

from skimage.feature import canny
from skimage.transform import hough_line
from skimage.transform import hough_line_peaks

# Local imports
from imageproc import ImageData
from pyradon.src.finder import Finder


class streakFinding:
    """
    Utilities for streak detection
    """

    @staticmethod
    def houghStreaks(
            image: ImageData,
            edgeSigma: float = 3.0,
            edgeLowThreshold: float | None = 10.,
            edgeHighThreshold: float | None = 20.
    ) -> dict:
        """
        Streak detection using Hough transform

        Parameters:
        -----------
        edgeSigma: float
            StDev of gaussian used for image smoothing before edge detection
        edgeLowThreshold: float
            Lower bound for hysteresis thresholding (edge linking)
        edgeHighThreshold: float
            Upper bound for hysteresis thresholding (edge linking)
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

        return houghResults
    
    @staticmethod
    def radonStreaks(
            image: ImageData,
            threshold: float,
            psf: float | np.ndarray,
            variance: float | np.ndarray,
            meanSubtract: bool = False,
            verbosity: int = 1,
    ):
        """
        Streak detection using Radon transform as described
        by Nir et al. 2018 and implemented in python here:
        https://github.com/guynir42/pyradon

        Nir et al. 2018: https://ui.adsabs.harvard.edu/abs/2018AJ....156..229N
        """

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
        print(f"Found {Nstreaks} streaks imimage")
        for i in range(Nstreaks):
            pyradonFinder.streaks[i].print()

        return
