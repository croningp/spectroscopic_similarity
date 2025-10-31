"""
Module for handling Raman spectroscopic data

.. moduleauthor:: Artem Leonov, Graham Keenan
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import PrivateAttr
from scipy import signal

from ....analysis import GenericSpectrum
from ....analysis.utils import interpolate_to_index

LASER_POWER = 785

MIN_X = 780
MAX_X = 1006


def _convert_wavelength_to_wavenumber(data):
    """Converts x from spectrometer to Raman shift in wavenumbers

    Arguments:
        data (iterable): Wavelength data to convert

    Returns:
        (:obj: np.array): Wavenumbers data
    """

    wavenumbers = [(10**7 / LASER_POWER) - (10**7 / wv) for wv in data]

    return np.array(wavenumbers)


class RamanSpectrum(GenericSpectrum):
    """Defines methods for Raman spectroscopic data handling

    Args:
        path(str, optional): Valid path to save the spectral data.
            If not provided, uses .//raman_data
    """

    x_label: str = "wavenumber"  # Raman shift in cm-1
    y_label: str = "intensity"  # Raman intensity in arbitrary units

    _original: NDArray = PrivateAttr()
    _reference: NDArray = PrivateAttr()

    def find_peaks_iteratively(self, limit=10, steps=100):
        """Finds all peaks iteratively moving the threshold

        Args:
            limit (int): Max number of peaks found at each iteration to stop
                after.
            steps (int): Max number of iterations.
        """

        gradient = np.linspace(self.y_data.max(), self.y_data.min(), steps)
        pl = [
            0,
        ]  # peaks length

        index = 0
        # Looking for peaks and decreasing height
        for index, peak_height in enumerate(gradient):
            peaks, _ = signal.find_peaks(self.y_data, height=peak_height)
            pl.append(len(peaks))
            diff = pl[-1] - pl[-2]
            if diff > limit:
                self.logger.debug(
                    "Stopped at iteration %s, with height %s, diff - %s",
                    index + 1,
                    peak_height,
                    diff,
                )
                break

        # Final peaks at previous iteration
        peaks, _ = signal.find_peaks(self.y_data, height=gradient[index - 1])

        # Updating widths
        pw = signal.peak_widths(self.y_data, peaks, rel_height=0.95)
        peak_xs = self.x_data.copy()[peaks][:, np.newaxis]
        peak_ys = self.y_data.copy()[peaks][:, np.newaxis]
        peaks_ids = np.around(peak_xs)
        peaks_left_ids = interpolate_to_index(self.x_data, pw[2])[:, np.newaxis]
        peaks_right_ids = interpolate_to_index(self.x_data, pw[3])[:, np.newaxis]

        # Packing for peak array
        self.peaks = np.hstack(
            (
                peaks_ids,
                peak_xs,
                peak_ys,
                peaks_left_ids,
                peaks_right_ids,
            )
        )

        return peaks_ids

    def set_as_reference(self):
        """Sets the current spectrum as a reference spectrum"""

        self._reference = self.y_data.copy()

    def load_spectrum(self, data_path, preprocessed=False):
        """Loads the spectral data"""

        # TODO: Implement loading from a file

        self._original = self.y_data

    def subtract_reference(self):
        """Subtracts reference spectrum and updates the current one"""

        if self._reference is None:
            raise ValueError("Please upload the reference first")

        self.y_data -= self._reference

    def default_processing(self):
        """Dummy method for quick processing. Returns spectral data!"""

        self.correct_baseline()
        self.smooth_spectrum()
        self.find_peaks_iteratively()

        return self.x_data, self.y_data, self.timestamp
