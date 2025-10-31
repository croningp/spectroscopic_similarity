import os

import numpy as np
import pytest

from AnalyticalLabware.analysis.bruker_spectrum import BrukerNMRSpectrum
from AnalyticalLabware.analysis.spinsolve_spectrum import SpinsolveNMRSpectrum

NMR_DATA_PATH = os.path.join("tests", "files", "nmr_data")


@pytest.mark.integration
def test_bruker_nmr_spectrum():
    """Test loading a Bruker NMR spectrum."""

    spectrum = BrukerNMRSpectrum()
    spectrum.load_spectrum(
        os.path.join(NMR_DATA_PATH, "test_data_bruker"), preprocessed=False
    )
    spectrum.default_processing()
    # I don't know why but for Bruker spectra you need to invert the x axis
    spectrum.x_data = spectrum.x_data[::-1]
    spectrum.autophase(function="acme")
    spectrum.find_peaks()

    assert spectrum.peaks is not None

    assert all(np.round(spectrum.peaks[:, 1].real, 2) == [1.62, 2.48, 3.48, 7.27])


@pytest.mark.integration
def test_magritek_nmr_spectrum():
    """Test loading a Magritek NMR spectrum."""

    spectrum = SpinsolveNMRSpectrum()
    spectrum.load_spectrum(
        os.path.join(NMR_DATA_PATH, "test_data_magritek"), preprocessed=False
    )
    spectrum.default_processing()
    spectrum.find_peaks()

    assert spectrum.peaks is not None

    assert all(
        np.round(spectrum.peaks[:, 1].real, 2)
        == [6.85, 3.02, 2.9, 2.38, 1.6, 0.81, 0.67]
    )
