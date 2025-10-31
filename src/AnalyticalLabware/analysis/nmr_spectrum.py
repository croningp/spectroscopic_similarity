import os
import queue
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import nmrglue as ng
import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from AnalyticalLabware.analysis.base_spectrum import GenericSpectrum
from AnalyticalLabware.analysis.spec_utils import (
    combine_map_to_regions,
    create_binary_peak_map,
    expand_regions,
    filter_noisy_regions,
    filter_regions,
    merge_regions,
)
from AnalyticalLabware.analysis.utils import find_nearest_value_index


@dataclass
class NMRData:
    chemical_conversion: queue.Queue = field(default_factory=queue.Queue)
    peak_ratio: queue.Queue = field(default_factory=queue.Queue)
    jaccard_index: queue.Queue = field(default_factory=queue.Queue)
    cross_correlation: queue.Queue = field(default_factory=queue.Queue)
    monitor_code: queue.Queue = field(default_factory=queue.Queue)


class NMRSpectrum(GenericSpectrum):
    """General class for handling NMR data

    Contains processing methods that are specific to NMR spectra"""

    x_label: str = "time"
    y_label: str = "intensity"

    udic: dict = ng.fileio.fileiobase.create_blank_udic(1)  # 1D spectrum
    processed_data: NMRData = NMRData()

    _data_path: str | None = None
    _uc: ng.fileio.fileiobase.unit_conversion = ng.fileio.fileiobase.uc_from_udic(udic)

    def show_spectrum(
        self,
        filename: str | None = None,
        title: str | None = None,
        label: str | None = None,
    ) -> None:
        """Plots the spectral data using matplotlib.pyplot module.

        Redefined from ancestor class to support axis inverting.

        Args:
            filename (str, optional): Filename for the current plot. If omitted,
                file is not saved.
            title (str, optional): Title for the spectrum plot. If omitted, no
                title is set.
            label (str, optional): Label for the spectrum plot. If omitted, uses
                the spectrum timestamp.
        """
        if label is None:
            label = f"{self.timestamp}"

        _, ax = plt.subplots(figsize=(12, 8))

        ax.plot(
            self.x_data,
            self.y_data.real,
            color="xkcd:navy blue",
            label=label,
        )

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

        if title is not None:
            ax.set_title(title)

        # plotting peaks if found
        if self.peaks is not None:
            plt.scatter(
                self.peaks[:, 1].real,
                self.peaks[:, 2].real,
                label="found peaks",
                color="xkcd:tangerine",
            )

        ax.legend()

        # inverting if ppm scale
        if self.x_label == "ppm":
            ax.invert_xaxis()

        if filename is None:
            plt.show()

        else:
            path = os.path.join(self.path, "images")
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path, f"{filename}.png"), dpi=150)

    def find_fid_start(self, threshold: float = 1.0) -> None:
        """On some spectrometers there is an instrument delay before the
        acquisition starts. If the leading zero values of the FID are not removed,
        the Fourier transformation will lead to oscillating baseline features that
        render the spectrum unusable. Therefore, this function cuts off the leading
        noise values of the FID."""

        # make sure the data is in time domain
        if self.x_label != "time":
            self.logger.error("Can only truncate FID in time domain.")
            return
        fid = self.y_data

        # Calculate noise level in the first 1% of the FID
        noise_std = np.std(fid[: len(fid) // 100])
        threshold = float(threshold * noise_std)

        # Find where signal starts (above noise threshold)
        start_idx = 0
        for i in range(len(fid) - 1):
            if np.abs(fid[i]) > threshold:
                start_idx = i
                break

        # Truncate the FID
        self.y_data: NDArray = fid[start_idx:]

        self.x_data = self.x_data[start_idx:]

        # updating udic and uc
        self.udic[0].update(size=self.x_data.size)
        self._uc = ng.fileio.fileiobase.uc_from_udic(self.udic)

    def fft(self, in_place: bool = True) -> None | NDArray:
        """Fourier transformation, NMR ordering of the results.

        This is the wrapper around nmrglue.process.proc_base.fft function.
        Please refer to original function documentation for details.

        Args:
            in_place(bool, optional): If True (default), self.y_data is updated;
                returns new array if False.

        Returns:
            Union[:np.array:, None]: If in_place is True, will return new array
                after FFT.
        """

        if in_place:
            self.y_data = ng.proc_base.fft(self.y_data)

            # updating x and y axis
            self.x_label = "ppm"
            self.x_data = self._uc.ppm_scale()

            # updating the udic to frequency domain
            self.udic[0]["time"] = False
            self.udic[0]["freq"] = True

        else:
            return ng.proc_base.fft(self.y_data)

    def autophase(
        self,
        in_place: bool = True,
        function: Literal["acme", "peak_minima"] = "peak_minima",
        p0: float = 0.0,
        p1: float = 0.0,
    ):
        """Automatic linear phase correction. FFT is performed!

        This is the wrapper around nmrglue.process.proc_autophase.autops
        function. Please refer to original function documentation for details.

        Args:
            in_place (bool, optional): If True (default), self.y_data is updated;
                returns new array if False.
            function (Union[str, Callable], optional): Algorithm to use for
                phase scoring. Builtin functions can be specified by one of the
                following strings: "acme" or "peak_minima". This refers to
                nmrglue.process.proc_autophase.autops function, "peak_minima"
                (default) was found to perform best.
            p0 (float, optional): Initial zero order phase in degrees.
            p1 (float, optional): Initial first order phase in degrees.

        Returns:
            Union[:np.array:, None]: If in_place is True, will return new array
                after phase correction.
        """

        # check if fft was performed
        if self.x_label == "time":
            self.fft()

        autophased: NDArray = ng.proc_autophase.autops(self.y_data, function, p0, p1, disp=False)  # type: ignore

        if in_place:
            self.y_data = autophased
            return

        return autophased

    def correct_baseline(self, in_place: bool = True, wd: int = 20) -> None | NDArray:  # type: ignore
        """Automatic baseline correction using distribution based
            classification method.

        Algorithm described in: Wang et al. Anal. Chem. 2013, 85, 1231-1239

        This is the wrapper around nmrglue.process.proc_bl.baseline_corrector
        function. Please refer to original function documentation for details.

        Args:
            in_place(bool, optional): If True (default), self.y_data is updated;
                returns new array if False.
            wd(int, optional): Median window size in pts.

        Returns:
            Union[:np.array:, None]: If in_place is True, will return new array
                after baseline correction.
        """

        # check if fft was performed
        if self.x_label == "time":
            self.fft()

        with np.testing.suppress_warnings() as sup:
            sup.filter(np.ComplexWarning, "")
            corrected = ng.proc_bl.baseline_corrector(self.y_data, wd)

        if in_place:
            self.y_data = corrected
            return

        return corrected

    def smooth_spectrum(self, in_place: bool = True, routine: Literal["ng", "savgol"] = "ng", **params) -> None | NDArray:  # type: ignore
        """Smoothes the spectrum.

        Depending on the routine chosen will use either Savitsky-Golay filter
        from scipy.signal module or nmrglue custom function.

        !Note: savgol will cast complex dtype to float!

        Args:
            in_place(bool, optional): If True (default), self.y_data is updated;
                returns new array if False.

            routine(str, optional): Smoothing routine.
                "ng" (default) will use custom smoothing function from
                    nmrglue.process.proc_base module.

                "savgol" will use savgol_filter method from scipt.signal module
                    defined in ancestor method.

            parram in params: Keyword arguments for the chosen routine function.
                For "savgol" routine:

                    window_length (int): The length of the filter window (i.e.
                        thenumber of coefficients). window_length must be a
                        positive odd integer.

                    polyorder (int): The order of the polynomial used to fit the
                            samples. polyorder must be less than window_length.

                For "ng" routine:

                    n (int): Size of smoothing windows (+/- points).

        Returns:
            Union[:np.array:, None]: If in_place is True, will return new array
                after baseline correction.
        """

        if routine == "savgol":
            super().smooth_spectrum(in_place=in_place, **params)

        elif routine == "ng":
            # using default value
            if not params:
                params = {"n": 5}

            if in_place:
                self.y_data = ng.proc_base.smo(self.y_data, **params)
                return

            return ng.proc_base.smo(self.y_data, **params)

        else:
            raise ValueError(
                'Please choose either nmrglue ("ng") or Savitsky-\
Golay ("savgol") smoothing routine'
            )

    def apodization(
        self,
        in_place: bool = True,
        function: Literal["em", "gm", "gmb"] = "em",
        **params,
    ) -> None | NDArray:
        """Applies a chosen window function.

        Args:
            in_place (bool, optional): If True (default), self.y_data is updated;
                returns new array if False.

            function (str, optional): Window function of choice.
                "em" - exponential multiply window (mimic NMRPipe EM function).
                "gm" - Lorentz-to-Gauss window function (NMRPipe GM function).
                "gmb" - Gauss-like window function (NMRPipe GMB function).

            param in params: Keyword arguments for the chosen window function:

                For "em":
                    See reference for nmrglue.proc_base.em and NMRPipe EM
                        functions.

                    lb (float): Exponential decay of the window in terms of a
                        line broadening in Hz. Negative values will generate an
                        increasing exponential window, which corresponds to a
                        line sharpening. The line-broadening parameter is often
                        selected to match the natural linewidth.

                For "gm":
                    See reference for nmrglue.proc_base.gm and NMRPipe GM
                        functions.

                    g1 (float): Specifies the inverse exponential to apply in
                        terms of a line sharpening in Hz. It is usually adjusted
                        to match the natural linewidth. The default value is
                        0.0, which means no exponential term will be applied,
                        and the window will be a pure Gaussian function.

                    g2 (float): Specifies the Gaussian to apply in terms of a
                        line broadening in Hz. It is usually adjusted to be
                        larger (x 1.25 - 4.0) than the line sharpening specified
                        by the g1 attribute.

                    g3 (float): Specifies the position of the Gaussian
                        function's maximum on the FID. It is specified as a
                        value ranging from 0.0 (Gaussian maximum at the first
                        point of the FID) to 1.0 (Gaussian maximum at the last
                        point of the FID). It most applications, the default
                        value of 0.0 is used.

                For "gmb":
                    See reference for nmrglue.proc_base.gmb and NMRPipe GMB
                        functions.

                    lb (float): Specifies an exponential factor in the chosen
                        Gauss window function. This value is usually specified
                        as a negative number which is about the same size as the
                        natural linewidth in Hz. The default value is 0.0, which
                        means no exponential term will be applied.

                    gb (float): Specifies a Gaussian factor gb, as used in the
                        chosen Gauss window function. It is usually specified as
                        a positive number which is a fraction of 1.0. The
                        default value is 0.0, which leads to an undefined window
                        function according to the formula; for this reason, the
                        Gaussian term is omitted from the calculation when gb
                        0.0 is given.

        Returns:
            Union[NDArray, None]: If in_place is True, will return new array
                after baseline correction.
        """
        # TODO check for Fourier transformation!

        if function == "em":
            # converting lb value to NMRPipe-like
            if "lb" in params:
                # deviding by spectral width in Hz
                params["lb"] = params["lb"] / self.udic[0]["sw"]

            if in_place:
                self.y_data = ng.proc_base.em(self.y_data, **params)
                return

            return ng.proc_base.em(self.y_data, **params)

        elif function == "gm":
            # converting values into NMRPipe-like
            if "g1" in params:
                params["g1"] = params["g1"] / self.udic[0]["sw"]

            if "g2" in params:
                params["g2"] = params["g2"] / self.udic[0]["sw"]

            if in_place:
                self.y_data = ng.proc_base.gm(self.y_data, **params)
                return

            return ng.proc_base.gm(self.y_data, **params)

        elif function == "gmb":
            # converting values into NMRPipe-like
            # for formula reference see documentation and source code of
            # nmrglue.proc_base.gmb function and NMRPipe GMB command reference
            if "lb" in params:
                param_a = np.pi * params["lb"] / self.udic[0]["sw"]
            else:
                param_a = 0.0

            if "gb" in params:
                param_b = -param_a / (2.0 * params["gb"] * self.udic[0]["size"])
            else:
                param_b = 0.0

            if in_place:
                self.y_data = ng.proc_base.gmb(self.y_data, a=param_a, b=param_b)
                return

            return ng.proc_base.gmb(self.y_data, a=param_a, b=param_b)

    def zero_fill(self, exponent: int = 1, in_place: bool = True) -> None | NDArray:
        """Zero filling the data by 2**n.

        Args:
            exponent (int): power of 2 to append 0 to the data.
            in_place (bool, optional): If True (default), self.y_data is updated;
                returns new array if False.

        Returns:
            Union[:np.array:, None]: If in_place is True, will return new array
                after baseline correction.
        """

        if in_place:
            # zero fill is useless when fft performed
            if self.x_label == "ppm":
                self.logger.warning(
                    "FFT already performed, zero filling \
skipped"
                )
                return

            # extending y axis
            self.y_data = ng.proc_base.zf_double(self.y_data, exponent)

            # extending x axis
            self.x_data: NDArray = np.linspace(
                self.x_data[0], self.x_data[-1] * 2**exponent, self.y_data.shape[0]
            )

            # updating udic and uc
            self.udic[0].update(size=self.x_data.size)
            self._uc = ng.fileio.fileiobase.uc_from_udic(self.udic)
            return

        return ng.proc_base.zf_double(self.y_data, exponent)

    def generate_peak_regions(
        self,
        magnitude=True,
        derivative=True,
        smoothed=True,
        d_merge=0.056,
        d_expand=0.0,
    ):
        """Generate regions if interest potentially containing compound peaks
            from the spectral data.

        Args:
            d_merge (float, Optional): arbitrary interval (in ppm!) to merge
                several regions, if the distance between is lower.
            d_expand (float, Optional): arbitrary value (in ppm!) to expand the
                regions after automatic assigning and filtering.

        Returns:
            :obj:np.array: 2D Mx2 array with peak regions indexes (rows) as left
                and right borders (columns).
        """

        # check if fft was performed
        if self.x_label != "ppm":
            self.logger.warning("Please perform FFT first.")
            return np.array([[]])

        # placeholders
        peak_map = np.full_like(self.x_data, False)
        magnitude_spectrum = self.y_data

        if magnitude:
            # looking for peaks in magnitude mode
            magnitude_spectrum: NDArray = np.sqrt(
                self.y_data.real**2 + self.y_data.imag**2
            )
            # mapping
            peak_map = np.logical_or(
                create_binary_peak_map(magnitude_spectrum), peak_map
            )
        else:
            peak_map = np.logical_or(create_binary_peak_map(self.y_data), peak_map)

        # additionally in the derivative
        if derivative:
            derivative_map = create_binary_peak_map(np.gradient(magnitude_spectrum))

            # combining
            peak_map = np.logical_or(derivative_map, peak_map)

        # and in the smoothed version
        if smoothed:
            # smoothing is only supported for the real part of the spectrum
            smoothed = scipy.ndimage.gaussian_filter1d(magnitude_spectrum.real, 3)

            # combining
            peak_map = np.logical_or(create_binary_peak_map(smoothed), peak_map)

        # extracting the regions from the full map
        regions = combine_map_to_regions(peak_map)

        # Skip further steps if no peaks identified
        if not regions.size > 0:
            return regions

        # filtering, merging, expanding
        regions = filter_regions(self.x_data, regions)
        regions = filter_noisy_regions(self.y_data, regions)
        if d_merge:
            regions = merge_regions(self.x_data, regions, d_merge=d_merge)
        if d_expand:
            regions = expand_regions(self.x_data, regions, d_expand=d_expand)

        return regions

    def default_processing(self) -> tuple[NDArray, NDArray, float | None]:
        """Default processing.

        Performs several processing methods with attributes chosen
        experimentally to achieve best results for the purpose of "fast",
        "reliable" and "reproducible" NMR analysis.
        """
        # truncate the FID if there is an instrument delay
        if self.y_data[0] == 0:
            self.find_fid_start()

        # TODO add processing for various nucleus
        if self.udic[0]["label"] in {"1H", "19F"}:
            self.apodization(function="gm", g1=1.2, g2=4.5)
            self.zero_fill()
            self.fft()
            if self.udic[0]["label"] == "19F":
                self.correct_baseline()
                self.autophase(function="acme")
                self.correct_baseline()
                self.highpass_filter(threshold=1000)

        return super().default_processing()

    def integrate_area(
        self, area: tuple[float, float], rule: Literal["trapz", "simps"] = "trapz"
    ) -> float:
        """Integrate the spectrum within given area.

        Redefined from ancestor method to discard imaginary part of the
        resulting integral.

        Args:
            area (Tuple[float, float]): Tuple with left and right border (X axis
                obviously) for the desired area.
            rule (str, optional): Method for integration, "trapz" - trapezoidal
                rule (default), "simps" - Simpson's rule.
        Returns:
            float: Definite integral within given area as approximated by given
                method.
        """

        result = super().integrate_area(area, rule)

        # discarding imaginary part and returning the absolute value
        # due to "NMR-order" of the x axis
        return abs(result.real)

    def integrate_regions(self, regions):
        """Integrate the given regions using nmrglue integration method.

        Check the corresponding documentation for details.

        Args:
            regions (:obj:np.array): 2D Mx2 array, containing left and right
                borders for the regions of interest, potentially containing
                peak areas (as found by self.generate_peak_regions method).

        Return:
            result (:obj:np.array): 1D M-size array contaning integration for
                each region of interest.
        """

        result = ng.analysis.integration.integrate(
            data=self.y_data,
            unit_conv=self._uc,
            limits=self.x_data[regions],  # directly get the ppm values
        )

        # discarding imaginary part
        return np.abs(np.real(result))

    def reference_spectrum(
        self,
        new_position: float,
        reference: Literal["highest", "closest"] | float = "highest",
    ) -> None:
        """Shifts the spectrum x axis according to the new reference.

        If old reference is omitted will shift the spectrum according to the
        highest peak.

        Args:
            new_position (float): The position to shift the peak to.
            reference (str): The current position of the reference
                peak or it's indication for shifting: either "highest" (default)
                or "closest" for selecting highest or closest to the new
                reference peak for shifting.
        """

        numeric_reference = 0.0
        # find reference if not given
        if isinstance(reference, str):
            if reference == "highest":
                # Looking for highest point
                numeric_reference = self.x_data[np.argmax(self.y_data)]
            elif reference == "closest":
                # Looking for closest peak among found across whole spectrum
                # Specifying area not to update self.peaks
                peaks = self.find_peaks(area=(self.x_data.min(), self.x_data.max()))
                # x coordinate
                peaks_xs = peaks[:, 1].real
                numeric_reference = peaks[np.argmin(np.abs(peaks_xs - new_position))][
                    1
                ].real
            else:
                self.logger.error(
                    'Please use either "highest" or "closest" reference, or give exact value.'
                )
                return
        else:
            # if reference is given as a float, use it
            numeric_reference = reference

        new_position, _ = find_nearest_value_index(self.x_data, new_position)

        diff = new_position - numeric_reference

        # shifting the axis
        self.x_data = self.x_data + diff

        # if peaks are recorded, find new
        if self.peaks is not None:
            self.find_peaks()

    def highpass_filter(self, in_place=True, threshold: float = 0.0):
        spectrum = self if in_place else deepcopy(self)
        spectrum.y_data = np.where(
            spectrum.y_data.real > threshold, spectrum.y_data.real, 0
        )
        return spectrum

    def cut(self, low_ppm: float, high_ppm: float, in_place=True):
        spectrum = self if in_place else deepcopy(self)
        selector = (spectrum.x_data < low_ppm) | (spectrum.x_data > high_ppm)
        spectrum.x_data = spectrum.x_data[selector]
        spectrum.y_data = spectrum.y_data[selector]
        return spectrum

    def trim_spectrum(self, low_ppm: float, high_ppm: float, in_place=True):
        spectrum = self if in_place else deepcopy(self)
        selector = (spectrum.x_data >= low_ppm) & (spectrum.x_data <= high_ppm)
        spectrum.x_data = spectrum.x_data[selector]
        spectrum.y_data = spectrum.y_data[selector]
        return spectrum

    def get_peak_ratio(
        self, peak1: tuple[float, float], peak2: tuple[float, float]
    ) -> float:
        """Calculates the ratio of two peaks in the spectrum.

        Args:
            peak1 (tuple[float, float]): Tuple with left and right border
                (X axis) for the first peak.
            peak2 (tuple[float, float]): Tuple with left and right border
                (X axis) for the second peak.

        Returns:
            float: Ratio of the peak areas.
        """

        # integration of the peaks
        area1 = self.integrate_area(peak1)
        area2 = self.integrate_area(peak2)

        # Suppress division by zero warning
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.divide(area1, area2)

        self.processed_data.peak_ratio.put(result)
        return float(result)

    def get_chemical_conversion(
        self, product_peak: tuple[float, float], substrate_peak: tuple[float, float]
    ) -> float:
        """Calculates the chemical conversion of the substrate to the product.

        Args:
            product_peak (tuple[float, float]): Tuple with left and right border
                (X axis) for the product peak.
            substrate_peak (tuple[float, float]): Tuple with left and right border
                (X axis) for the substrate peak.

        Returns:
            float: Chemical conversion of the substrate to the product.
        """

        # integration of the peaks
        area_product = self.integrate_area(product_peak)
        area_substrate = self.integrate_area(substrate_peak)

        # Suppress division by zero warning
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.divide(area_product, area_substrate + area_product)

        self.processed_data.chemical_conversion.put(result)
        return float(result)
