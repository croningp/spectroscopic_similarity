"""Base module for interfacing with OceanOptics Spectrometers

.. moduleauthor:: Artem Leonov, Graham Keenan
"""

import time

import numpy as np

from AnalyticalLabware.core.logging.loggers import get_logger

# Spectrometer Types:
SPECS = {"UV": "2192", "RAMAN": "QE-PRO", "IR": "NIRQUEST"}


class UnsupportedSpectrometer(Exception):
    """Exception for unsupported spectrometer types"""


class NoSpectrometerDetected(Exception):
    """Exception for when no spectrometer is detected"""


class OceanOpticsSpectrometer:
    """Base class for interfacing with OceanOptics Spectrometers"""

    def __init__(self, spec_type, name=None):
        """
        Args:
            spec_type (str): The type of spectrometer, e.g. 'IR', 'raman', etc.
            name (str, optional): Device name for easier access
        """
        # class level import to allow AnalyticalLabware to be imported without seabreeze
        import seabreeze

        seabreeze.use("cseabreeze")
        import seabreeze.spectrometers as sb  # noqa: E402

        self.integration_time = 0.01  # in seconds
        self.device_list = sb.list_devices()
        self.__spec = self._get_spectrometer(spec_type, self.device_list)
        self._spectrometer = sb.Spectrometer(self.__spec)
        self.name = name
        self._delay = 0.01

        self.logger = get_logger(f"{self}")

        self.set_integration_time(self.integration_time)

    def __str__(self) -> str:
        return type(self).__name__

    def set_integration_time(self, integration_time):
        """Sets the integration time for the spectrometer

        Args:
            integration_time (float): Desired integration time in seconds!
        """

        self._spectrometer.open()
        self.integration_time = integration_time
        integration_time *= 1000 * 1000  # converting to microseconds
        self.logger.debug(
            "Setting the integration time to %s microseconds", integration_time
        )
        self._spectrometer.integration_time_micros(integration_time)
        self._spectrometer.close()

    def scan(self, num_scans=3):
        """Reads the spectrometer and returns the spectrum

        Args:
            num_scans (int, opitonal): Number of 'scans'

        Returns:
            (Tuple): Tuple containing spectrum wavelengths and intensities as numpy arrays
                Example: (array(wavelengths), array(intensities))
        """

        i_mean = []
        self.logger.debug("Scanning")
        self._spectrometer.open()
        for _i in range(num_scans):
            wavelengths, intensities = self._spectrometer.spectrum()
            i_mean.append(intensities)
            time.sleep(self._delay)

        intensities = np.mean(i_mean, axis=0)
        self._spectrometer.close()
        return (wavelengths, intensities)

    def _get_spectrometer(self, spec_type: str, devices: list) -> str:
        """Gets the Spectrometer from Seabreeze that matches given type

        Arguments:
            spec_type {str} -- Type of spectrometer to look for

        Raises:
            UnsupportedSpectrometer -- If the spec_type is not present

        Returns:
            str -- Name of the spectrometer
        """

        if not devices:
            raise NoSpectrometerDetected("Are the spectrometers plugged in?")
        if spec_type in SPECS.keys():
            for dev in devices:
                if SPECS[spec_type] in str(dev):
                    return dev
        raise UnsupportedSpectrometer(f"Spectrometer {spec_type} unsupported!")
