"""Script used to test-run the image processing routines"""

# Add src directory to path
import sys
sys.path.insert(0, "./src")

from process_image import ImageIO
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
image1 = ImageIO.load(fitsFile1)
image2 = ImageIO.load(fitsFile2)


# Take image difference and create new ImageData object
diffimPath = f"{DIRS['data']}/diffim.fits"
diffimage = ImageIO.create(
    data=image1.data-image2.data,
    header=image2.header,
    save_to=diffimPath
)