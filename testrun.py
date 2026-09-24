"""Script used to test-run the image processing routines"""

from process_image import load_image
from astropy.visualization import PercentileInterval

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
image1 = load_image(fitsFile1)
image2 = load_image(fitsFile2)

