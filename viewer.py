"""Minimal napari FITS browser with Previous / Next buttons.

Usage:
    python fits_browser.py path/to/folder
    python fits_browser.py a.fits b.fits c.fits

Requires: pip install "napari[all]" astropy magicgui numpy
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, "./src")

# For linux platforms using wayland
if sys.platform.startswith("linux") and os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

import napari
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from magicgui.widgets import (
    CheckBox,
    Container,
    FloatSpinBox,
    Label,
    PushButton,
    SpinBox,
)
from napari.utils.notifications import show_error, show_info


def find_files(args):
    """Turn command-line arguments (files and/or folders) into a sorted file list."""
    fitsSuffixes = [".fits", ".fit", ".fts", ".fz"]
    files = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            files += [f for f in p.iterdir() if f.suffix.lower() in fitsSuffixes]
        elif p.is_file():
            files.append(p)
    return sorted(files)


def load_fits(path):
    """Return the first 2D image found in the file as a float32 array."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = np.squeeze(hdu.data)
            if data.ndim == 2:
                return data.astype(np.float32)
    raise ValueError(f"No 2D image data found in {path}")


# ---------------------------------------------------------------------------
# Detector adapters. Each takes the 2D image plus keyword parameters and returns
# a list of line segments (x0, y0, x1, y1), in image pixel coordinates.
# ---------------------------------------------------------------------------

def houghDetector(
        image, 
        edge_low_threshold, 
        edge_high_threshold, 
        edge_sigma, 
        hough_line_length, 
        hough_line_gap,
        hough_threshold,
        angle_steps
    ):
    """
    Hough Transform streak detector (Canny + probabilistic Hough)
    """
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    # Normalize the image
    lo, hi = np.nanpercentile(image, [1, 99.5])
    norm = np.nan_to_num(np.clip((image - lo) / (hi - lo + 1e-12), 0, 1))

    # Detect edges
    edges = canny(
        norm, 
        sigma=edge_sigma, 
        low_threshold=edge_low_threshold, 
        high_threshold=edge_high_threshold
    )

    # Perform Hough transform to detect straight-line streaks
    angles = np.linspace(-np.pi/2, np.pi/2, angle_steps, endpoint=False)
    lines = probabilistic_hough_line(
        edges, 
        threshold=hough_threshold, 
        line_length=int(hough_line_length), 
        line_gap=int(hough_line_gap),
        theta=angles
    )

    # Return streak starting/ending points
    streakEndpoints = [
        (x0, y0, x1, y1) for (x0, y0), (x1, y1) in lines
    ]

    return streakEndpoints


def pyradonDetector(
        image, 
        psf, 
        threshold
    ):
    """
    Streak detection using Radon transform as described
    by Nir et al. 2018 and implemented in python here:
    https://github.com/guynir42/pyradon

    Nir et al. 2018: https://ui.adsabs.harvard.edu/abs/2018AJ....156..229N

    Parameters:
    -----------
    image: np.ndarray
        Image to perform streak detection on
    psf:
        Scalar PSF FWHM (pixels)
    threshold: float
        S/N threshold for detected streaks

    Returns:
    --------
    streakEndpoints: list
        List of streak endpoints (x0, y0, x1, y1)
    """

    # Local import
    from pyradon.finder import Finder
    
    # Measure image variance
    _,_,imgStd = sigma_clipped_stats(
        image, sigma=5.0
    )

    # Setup the pyradon streak finder
    pyradonFinder = Finder(
        verbosity=1, 
        threshold=threshold,
        use_exclude=False,
        use_subtract_mean=True
    )

    # Provide input data to finder
    pyradonFinder.input(
        image,
        variance=imgStd**2,
        psf=psf
    )

    # Return streak starting/ending points
    streakEndpoints = [
        (s.x1, s.y1, s.x2, s.y2) for s in pyradonFinder.streaks
    ]

    return streakEndpoints


class FitsBrowser:

    def __init__(self, files):

        self.files = files
        self.index = 0
        self.annotations = {}     # path -> (shape data, shape types, edge widths, edge colors)
        self.current_path = None  # image currently displayed

        self.viewer = napari.Viewer()
        first = load_fits(files[0])
        self.layer = self.viewer.add_image(first, name="image")

        # Add a shape layer for streak labeling. Default tool = "add_polyline"
        self.shapes_layer = self.viewer.add_shapes(
            name="streaks",
            ndim=2,
            edge_width=8,
            edge_color="green",
            face_color="transparent",
            opacity=0.3,
        )
        self.shapes_layer.mode = "add_polyline"
        self.viewer.layers.selection.active = self.shapes_layer

        # Setup the image navigation control panel
        self.label = Label(value="")
        prev_btn = PushButton(text="◀ Previous")
        next_btn = PushButton(text="Next ▶")
        prev_btn.changed.connect(lambda: self.step(-1))
        next_btn.changed.connect(lambda: self.step(+1))
        panel = Container(
            widgets=[self.label, prev_btn, next_btn]
        )
        self.viewer.window.add_dock_widget(panel, area="right", name="Images")

        self.show(0)


    def show(self, index):
        """
        Function that displays the image indicated by index
        and its associated streak mask (shape layer).
        """

        # Get the current shape layer
        layer = self.shapes_layer

        # Stash the shapes of the image we're leaving.
        if self.current_path is not None:
            self.annotations[self.current_path] = (
                [np.array(s) for s in layer.data],
                list(layer.shape_type),
                list(layer.edge_width),
                np.array(layer.edge_color),
            )

        # Set path/index/data for the new image being displayed
        self.index = index % len(self.files)
        path = self.files[self.index]
        data = load_fits(path)
        self.current_path = path

        # Clear the shape layer, then restore this image's shapes (if any).
        layer.selected_data = set(range(layer.nshapes))
        layer.remove_selected()
        saved = self.annotations.get(path)
        if saved and saved[0]:
            shapes, types, widths, colors = saved
            layer.add(shapes, shape_type=types, edge_width=widths, edge_color=colors)

        # Set the display range from percentiles so faint streaks are visible.
        lo, hi = np.nanpercentile(data, [1, 99.5])
        self.layer.data = data
        self.layer.contrast_limits_range = (float(np.nanmin(data)), float(np.nanmax(data)))
        self.layer.contrast_limits = (float(lo), float(hi))
        self.layer.name = path.name
        
        self.label.value = f"{self.index + 1} / {len(self.files)}\n{path.name}"
        self.viewer.title = path.name

    def add_detector(self, name, detect_func, params, color="red"):
        """Add a dock panel with parameter fields and a Run button for a detector.

        detect_func(image, **params) -> list of (x0, y0, x1, y1) segments.
        params maps each argument name to a dict with value/min/max/step. An int
        `value` gives an integer field; a float gives a decimal field.
        color is the edge color used for this detector's lines.
        """
        fields = {}
        for pname, spec in params.items():
            if isinstance(spec["value"], int):
                w = SpinBox(value=spec["value"], min=spec.get("min", 0),
                            max=spec.get("max", 10_000), step=spec.get("step", 1), label=pname)
            else:
                w = FloatSpinBox(value=spec["value"], min=spec.get("min", 0.0),
                                 max=spec.get("max", 1e6), step=spec.get("step", 0.01), label=pname)
            fields[pname] = w

        replace = CheckBox(value=True, text="Replace existing shapes")
        run = PushButton(text=f"Run {name}")

        def on_run():
            """
            Run streak detection and add segments to shape layer
            """
            kwargs = {p: w.value for p, w in fields.items()}
            try:
                segments = detect_func(self.layer.data, **kwargs)
            except Exception as exc:
                show_error(f"{name} failed: {exc}")
                return
            self.add_segments(segments, color, replace=replace.value)
            show_info(f"{name}: {len(segments)} line(s) found")

        run.changed.connect(on_run)
        panel = Container(widgets=[*fields.values(), replace, run])
        self.viewer.window.add_dock_widget(panel, area="right", name=name)


    def add_segments(self, segments, color="red", replace=True):
        """
        Draw (x0, y0, x1, y1) segments on the shapes layer (napari uses row, col).
        """
        layer = self.shapes_layer
        if replace:
            layer.selected_data = set(range(layer.nshapes))
            layer.remove_selected()
        if len(segments):
            lines = [np.array([[y0, x0], [y1, x1]]) for x0, y0, x1, y1 in segments]
            layer.add(
                lines, 
                shape_type="line",
                edge_width=layer.current_edge_width, 
                edge_color=color
            )


    def step(self, delta):
        """
        Function called by previous and next buttons
        """
        self.show(self.index + delta)


if __name__ == "__main__":
    files = find_files(sys.argv[1:])
    if not files:
        sys.exit("No FITS files found. Usage: python fits_browser.py <folder or files>")
    browser = FitsBrowser(files)

    # Register detectors: a name, an adapter function, and its parameter fields.
    browser.add_detector(
        "Hough Detector",
        houghDetector,
        {
            "edge_low_threshold": dict(value=0.05, min=0.0, max=1.0, step=0.01),
            "edge_high_threshold": dict(value=0.20, min=0.0, max=1.0, step=0.01),
            "edge_sigma": dict(value=2.0, min=0.0, max=20.0, step=0.5),
            "hough_line_length": dict(value=100, min=1, max=20_000),
            "hough_line_gap": dict(value=10, min=0, max=1_000),
            "hough_threshold":dict(value=100, min=5, max=1_000, step=5),
            "angle_steps":dict(value=360, min=30, max=720, step=30)
        },
        color="red",
    )
    browser.add_detector(
        "PyRadon Detector",
        pyradonDetector,
        {
            "psf":dict(value=3.0, min=1.0, max=10.0, step=0.25), 
            "threshold":dict(value=10.0, min=1.0, max=100.0, step=1.0)
        },
        color="blue",
    )

    # Run napari
    napari.run()