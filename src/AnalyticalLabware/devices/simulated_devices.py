from __future__ import annotations

import time

import numpy as np

from AnalyticalLabware.analysis.base_spectrum import GenericSpectrum
from AnalyticalLabware.analysis.spinsolve_spectrum import SpinsolveNMRSpectrum
from AnalyticalLabware.core.logging.loggers import get_logger

# ### Simulated devices ###


class _SimulatedSpectrum(GenericSpectrum):
    def __init__(self, *args, **kwargs):
        super().__init__(path=False)

    def load_spectrum(self, *args, **kwargs):
        x_values = np.linspace(-100, 100, 1000)
        y_values = 1 / (1 + x_values**2)
        super().load_data(x_values=x_values, y_values=y_values, timestamp=time.time())

    def save_pickle(self, *args, **kwargs):
        pass

    def default_processing(self, *args, **kwargs):
        return self.x_data, self.y_data, 42.0


class _SimulatedNMRSpectrum(SpinsolveNMRSpectrum):
    def __init__(self, *args, **kwargs):
        self.rng: np.random.Generator = np.random.default_rng(42)
        super().__init__(path=False)

    def load_spectrum(self, *args, **kwargs):
        """Simulate spectrum with random number of peaks"""
        n_peaks = kwargs.get("n_peaks", self.rng.integers(5, 15))
        # Generating random spectrum
        positions = np.linspace(0, -10, 10000)
        intensity_values = np.zeros_like(positions)
        for _ in range(n_peaks):
            # Peak center, random within [-7.5, -2.5)
            p0 = self.rng.random() * -5 - 2.5
            # Peak FWHM, random within [0.01, 0.03)
            peak_width = self.rng.random() * 0.02 + 0.01
            # Using Lorentzian line shape function
            ppm_values = (positions - p0) / (peak_width / 2)
            intensity_values += 1 / (1 + ppm_values**2)
        # Adding some Gaussian noise
        intensity_values += self.rng.normal(0, 0.01, intensity_values.size)
        # Assigning
        self.x_data = positions[::-1]
        self.y_data = intensity_values
        # Changing axis mapping
        self.x_label = "ppm"
        # Channel used in analysis, so setting to ''
        # TODO Add additional parameters if needed!
        self.parameters = {"rxChannel": ""}
        # Setting datapath to pretend that spectrum was loaded from file
        self.data_path = "dummy_path"

    def integrate_regions(self, regions):
        # Not working if no unit conversion attribute is set.
        raise NotImplementedError


class SimOceanOpticsRaman:
    def __init__(self, *args, **kwargs):
        self.logger = get_logger(f"{self}")
        self.spectrum = _SimulatedSpectrum()

    def get_spectrum(self):
        self.spectrum.load_spectrum()

    def obtain_reference_spectrum(self):
        pass
