"""Module for NMR spectral data loading and manipulating"""

# pylint: disable=attribute-defined-outside-init

import os
import time
from typing import Any

import nmrglue as ng
import numpy as np

from AnalyticalLabware.analysis.nmr_spectrum import NMRSpectrum

# standard filenames for spectral data
FID_DATA = "data.1d"
ACQUISITION_PARAMETERS = "acqu.par"
PROCESSED_SPECTRUM = "spectrum_processed.1d"  # not always present
PROTOCOL_PARAMETERS = "protocol.par"

# format used in acquisition parameters
TIME_FORMAT = r"%Y-%m-%dT%H:%M:%S.%f"

# reserved for future use
JCAMP_DX_SPECTRUM = "nmr_fid.dx"
CSV_SPECTRUM = "spectrum.csv"

# filename for shimming parameters
SHIMMING_PARAMETERS = "shim.par"
SHIMMING_FID = "sample_fid.1d"
SHIMMING_SPECTRUM = "spectrum.1d"


class SpinsolveNMRSpectrum(NMRSpectrum):
    """Class for Spinsolve NMR spectrum loading and handling.

    Contains custom methods for loading the proprietary Spinsolve format into something
    that is compatible with nmrglue."""

    last_shimming_time: time.struct_time | None = None
    last_shimming_results: dict | None = None

    def load_spectrum(self, data_path, preprocessed=False):
        """Loads the spectra from the given folder.

        If preprocessed argument is True, loading the spectral data already
        processed by the Spinsolve software (fft + autophasing).

        Args:
            data_path (str): Path where NMR data has been saved.
            preprocessed (bool, optional): If True - will load preprocessed (by
                Spinsolve software) spectral data. If False (default) - base fid
                is loaded and used for further processing.
        """

        # filepaths
        param_path = os.path.join(data_path, ACQUISITION_PARAMETERS)
        processed_path = os.path.join(data_path, PROCESSED_SPECTRUM)
        fid_path = os.path.join(data_path, FID_DATA)

        # loading parameters
        try:
            parameters = self.extract_parameters(param_path)
        except FileNotFoundError:
            # this happens only if shimming was performed
            shim_path = os.path.join(data_path, SHIMMING_PARAMETERS)
            parameters = self.extract_parameters(shim_path)

            # updating placeholders
            self.last_shimming_results = {
                parameter: parameters[parameter]
                for parameter in parameters
                if parameter.startswith("shim")
            }

            # updating last shimming time
            self.last_shimming_time = time.strptime(
                parameters["CurrentTime"], TIME_FORMAT
            )

            # updating file names for the shimming
            processed_path = os.path.join(data_path, SHIMMING_SPECTRUM)

            # forcing preprocessed to deal with frequency domain not raw FID
            preprocessed = True

        self.data_path = data_path

        # extracting the time from acquisition parameters
        spectrum_time = time.strptime(parameters["CurrentTime"], TIME_FORMAT)

        timestamp = round(time.mktime(spectrum_time))

        # loading raw fid data
        if not preprocessed:
            x_axis, y_real, y_img = self.extract_data(fid_path)
            spectrum_data = np.array(y_real + y_img * 1j)

            # updating the universal dictionary
            self.udic[0].update(
                # spectral width in kHz
                sw=parameters["bandwidth"] * 1e3,
                # carrier frequency
                car=parameters["bandwidth"] * 1e3 / 2 + parameters["lowestFrequency"],
                # observed frequency
                obs=parameters["b1Freq"],
                # number of points
                size=parameters["nrPnts"],
                # time domain
                time=True,
                # label, e.g. 1H
                label=parameters["rxChannel"],
            )

            # changing axis label according to raw FID
            self.x_label = "time"

            # creating unit conversion object
            self._uc = ng.fileio.fileiobase.uc_from_udic(self.udic)

        # check for the preprocessed file, as it's not always present
        elif os.path.isfile(processed_path) and preprocessed:
            # loading frequency axis and real part of the complex spectrum data
            x_axis, spectrum_data, _ = self.extract_data(processed_path)
            # reversing spectrum order to match default nmr order
            # i.e. highest - left
            x_axis = x_axis[::-1]
            spectrum_data = spectrum_data[::-1]
            # updating axis label
            self.x_label = "ppm"

        else:
            self.logger.warning(
                "It seems that a raw FID data was loaded. Please set the preprocessed "
                "argument to False if you want to load the raw data."
            )
            raise AttributeError(
                f"Processed spectrum was not found in the \
supplied directory <{data_path}>"
            )

        # loading all data
        super().load_data(x_axis, spectrum_data, int(timestamp))

    # ### PUBLIC METHODS TO LOAD RAW DATA ###

    def extract_data(self, spectrum_path):
        """Reads the Spinsolve spectrum file and extracts the spectrum data
        from it.

        Data is stored in binary format as C struct data type. First 32 bytes
        (8 integers * 4 bytes) contains the header information and can be
        discarded. The rest data is stored as float (4 byte) data points for X
        axis and complex number (as float, float for real and imaginary parts)
        data points for Y axis.

        Refer to the software manual (v 1.16, July 2019 Magritek) for details.

        Args:
            spectrum_path: Path to saved NMR spectrum data.

        Returns:
            tuple[x_axis, y_data_real, y_data_img]:
                x_axis (:obj: np.array, dtype='float32'): X axis points.
                y_data_real (:obj: np.array, dtype='float32'): real part of the
                    complex spectrum data.
                y_data_img (:obj: np.array, dtype='float32'): imaginary part of
                    the complex spectrum data.

        """

        # reading data with numpy fromfile
        # the header is discarded
        spectrum_data = np.fromfile(spectrum_path, dtype="<f")[8:]

        x_axis = spectrum_data[: len(spectrum_data) // 3]

        # breaking the rest of the data into real and imaginary part
        y_real = spectrum_data[len(spectrum_data) // 3 :][::2]
        y_img = spectrum_data[len(spectrum_data) // 3 :][1::2]

        return (x_axis, y_real, y_img)

    def extract_parameters(self, params_path: str) -> dict[str, Any]:
        """Get the NMR parameters from the given folder.

        Args:
            params_path (str): Path to saved NMR data.

        Returns:
            Dict: Acquisition parameters.
        """

        # loading spectrum parameters
        spec_params = {}
        with open(params_path) as fileobj:
            param = fileobj.readline()
            while param:
                # in form of "Param"       = "Value"\n
                parameter, value = param.split("=", maxsplit=1)
                # stripping from whitespaces, newlines and extra doublequotes
                parameter = parameter.strip()
                value = value.strip(' \n"')
                # special case: userData
                # converting to nested dict
                if parameter == "userData" and value:
                    values = value.split(";")
                    value = {}
                    for key_value in values:
                        key, param_value = key_value.split("=")
                        value[key] = param_value
                # converting values to float if possible, but skip dicts
                try:
                    if not isinstance(value, dict):
                        spec_params[parameter] = float(value)
                    else:
                        spec_params[parameter] = value
                except (ValueError, TypeError):
                    spec_params[parameter] = value
                param = fileobj.readline()

        return spec_params

    @classmethod
    def from_pickle(cls, path: str):
        # overwritten from abstract class to allow updating of unit conversion
        new_spectrum = super().from_pickle(path)
        new_spectrum._uc = ng.fileio.fileiobase.uc_from_udic(new_spectrum.udic)

        # updating axis mapping from "time" default
        if new_spectrum.udic[0]["freq"]:
            new_spectrum.x_label = "ppm"

        elif new_spectrum.udic[0]["time"]:
            new_spectrum.x_label = "time"

        return new_spectrum
