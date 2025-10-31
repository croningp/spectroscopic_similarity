import logging
import os
import pickle  # nosec B403
from dataclasses import asdict, dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import integrate, signal, sparse

from AnalyticalLabware.core.logging.loggers import get_logger

from .utils import find_nearest_value_index, interpolate_to_index


@dataclass
class GenericSpectrum:
    """General class for handling spectroscopic data

    Contains methods for data manipulation (load/save) and basic processing
    features, such as baseline correction, smoothing, peak picking and
    integration.

    All data processing happens in place!
    """

    # model_config = ConfigDict(
    #     arbitrary_types_allowed=True
    # )

    path: str = "."

    x_data: NDArray = field(default_factory=lambda: np.array([]))
    y_data: NDArray = field(default_factory=lambda: np.array([]))
    # x_data: NDArray = np.array([])
    # y_data: NDArray = np.array([])
    # labels for plotting (also relevant to distinguish between time and frequency
    # domains when performing Fourier transformation)
    x_label: str = "X axis"
    y_label: str = "Y axis"
    peaks: NDArray | None = None
    timestamp: float | None = None

    # internal properties
    _baseline: NDArray | None = None

    @property
    def logger(self) -> logging.Logger:
        return get_logger(f"{self.__class__.__name__}")

    def load_spectrum(self, data_path: str, preprocessed: bool):
        """Loads the spectral data from the given path.

        This method must be redefined in child classes.

        Args:
            data_path (str): Path to the spectral data file.
            preprocessed (bool): If True, loads preprocessed data, otherwise
                loads raw data.
        """
        raise NotImplementedError(
            "This method must be implemented in a subclass to load the spectrum data."
        )

    def load_data(self, x_values: NDArray, y_values: NDArray, timestamp):
        """Loads the spectral data.

        This method must be redefined in ancestor classes.

        Args:
            x (:obj: np.array): An array with data to be plotted as "x" axis.
            y (:obj: np.array): An array with data to be plotted as "y" axis.
            timestamp (float): Timestamp to the corresponding spectrum.
        """

        if x_values.shape != y_values.shape:
            raise ValueError("X and Y data must have same dimension.") from None

        self.x_data = x_values
        self.y_data = y_values
        self.timestamp = timestamp

    def save_pickle(self, filename: str | None = None):
        """Saves the data to given path using python pickle module.

        Args:
            filename (str, optional): Filename for the current spectrum. If
                omitted, using current timestamp.
            verbose (bool, optional): If all processed data needs to be saved as
                well. Default: False.
        """
        if filename is None:
            filename = f"{self.timestamp}.pkl"
        else:
            if not filename.endswith(".pkl"):
                filename += ".pkl"

        path = os.path.join(self.path, filename)

        # # data = self.model_dump()
        data = asdict(self)

        # # Ensure all numpy arrays are converted to lists
        # def convert_np_array(obj):
        #     if isinstance(obj, np.ndarray):
        #         if np.iscomplexobj(obj):
        #             cmplx_list:list[complex] = obj.tolist()
        #             return [{"real": x.real, "imag": x.imag} for x in cmplx_list]
        #         else:
        #             return obj.tolist()
        #     return obj

        # data = {key: convert_np_array(value) for key, value in data.items()}

        # with open(path, "w") as f:
        #     json.dump(data, f)

        with open(path, "wb") as f:
            pickle.dump(data, f)

        self.logger.info("Saved in %s", path)

    @classmethod
    def from_pickle(cls, path: str):
        """Loads model from a pickle file and returns a new instance."""
        # with open(path) as f:
        #     data:dict = json.load(f)

        # def restore_np_array(obj):
        #     """Restores numpy arrays from lists."""
        #     if isinstance(obj, list):
        #         if all(isinstance(x, dict) for x in obj):
        #             # If it's a list of dicts, assume complex numbers
        #             return np.array([complex(x["real"], x["imag"]) for x in obj])
        #         else:
        #             # Otherwise, assume it's a regular list
        #             return np.array(obj)
        #     return obj

        # data = {key: restore_np_array(value) for key, value in data.items()}

        # return cls(**data)
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa DUO103 # nosec B301

        return cls(**data)

    def trim(self, xmin: float, xmax: float, in_place=True) -> tuple[NDArray, NDArray]:
        """Trims the spectrum data within specific X region

        Args:
            xmin (float): Minimum position on the X axis to start from.
            xmax (float): Maximum position on the X axis to end to.
            in_place (bool): If trimming happens in place, else returns new
                array as trimmed copy.

        Returns:
            (tuple[NDArray, NDArray]): Trimmed copy of the original array as
                tuple with X and Y points respectively.
        """

        # Creating the mask to map arrays
        above_ind = self.x_data > xmin
        below_ind = self.x_data < xmax
        full_mask = np.logical_and(above_ind, below_ind)

        # Mapping arrays if they are supplied
        if in_place:
            self.y_data = self.y_data[full_mask]
            self.x_data = self.x_data[full_mask]
            if self._baseline is not None and self._baseline.shape == full_mask.shape:
                self._baseline = self._baseline[full_mask]

        return (self.x_data.copy()[full_mask], self.y_data.copy()[full_mask])

    def show_spectrum(
        self,
        filename=None,
        title=None,
        label=None,
    ):
        """Plots the spectral data using matplotlib.pyplot module.

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

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(
            self.x_data,
            self.y_data,
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
                self.peaks[:, 1],
                self.peaks[:, 2],
                label="found peaks",
                color="xkcd:tangerine",
            )

        ax.legend()

        if filename is None:
            plt.show()

        else:
            path = os.path.join(self.path, "images")
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path, f"{filename}.png"), dpi=150)

    def find_peaks(
        self, threshold=0.1, min_width=2, min_dist=None, area=None, decimals=1
    ) -> NDArray:
        """Finds all peaks above the threshold with at least min_width width.

        Args:
            threshold (float, optional): Relative peak height with respect to
                the highest peak.
            min_width (int, optional): Minimum peak width.
            min_dist (int, optional): Minimum distance between peaks.
            area (Tuple(int, int), optional): Area to search peaks in. Supplied
                as min, max X values tuple.

        Return:
            (:obj: NDArray): An array of peaks ids as rounded peak_x coordinate
                value. If searching within specified area, full peak information
                matrix is returned, see below for details.

        Also updates the self.peaks attrbiute (if "area" is omitted) as:
            (:obj: NDArray): An (n_peaks x 5) array with peak data as columns:
                peak_id (float): Rounded peak_x coordinate value.
                peak_x (float): X-coordinate for the peak.
                peak_y (float): Y-coordinate for the peak.
                peak_left_x (float): X-coordinate for the left peak border.
                peak_right_x (float): X-coordinate for the right peak border.

        Peak data is accessed with indexing, e.g.:
            self.peaks[n] will give all data for n's peak
            self.peaks[:, 2] will give Y coordinate for all found peaks
        """

        # trimming
        if area is not None:
            spec_y = self.trim(area[0], area[1], False)[1]
        else:
            spec_y = self.y_data.copy()

        threshold *= self.y_data.max() - self.y_data.min()

        with np.testing.suppress_warnings() as sup:
            sup.filter(np.ComplexWarning, "")
            peaks, _ = signal.find_peaks(
                spec_y, height=threshold, width=min_width, distance=min_dist
            )

            # obtaining width for full peak height
            # TODO deal with intersecting peaks!
            # TODO deal with incorrect peak width
            pw = signal.peak_widths(spec_y, peaks, rel_height=0.95)

        # convert peak indices to original array
        if area is not None:
            # find anchor index of first peak to deduce shift
            anchor_index = np.where(self.y_data == spec_y[peaks[0]])[0][0]
            peaks += anchor_index

        # converting all to column vectors by adding extra dimension along 2nd
        # axis. Check documentation on np.newaxis for details
        peak_xs = self.x_data.copy()[peaks][:, np.newaxis]
        peak_ys = self.y_data.copy()[peaks][:, np.newaxis]

        peaks_ids = np.around(peak_xs, decimals=decimals)
        peaks_left_ids = interpolate_to_index(self.x_data, pw[2])[:, np.newaxis]
        peaks_right_ids = interpolate_to_index(self.x_data, pw[3])[:, np.newaxis]

        if area is None:
            # updating only if area is not specified
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

        return np.hstack(
            (
                peaks_ids,
                peak_xs,
                peak_ys,
                peaks_left_ids,
                peaks_right_ids,
            )
        )

    def correct_baseline(self, lmbd=1e3, asymmetric_weight=0.01, n_iter=10):
        """Generates and subtracts the baseline for the given spectrum.

        Based on Eilers, P; Boelens, H. (2005): Baseline Correction with
            Asymmetric Least Squares Smoothing.

        Default values chosen arbitrary based on processing Raman spectra.

        Args:
            lmbd (float): Arbitrary parameter to define the smoothness of the
                baseline the larger lmbd is, the smoother baseline will be,
                recommended value between 1e2 and 1e5.
            p (float): An asymmetric least squares parameter to compute the
                weights of the residuals, chosen arbitrary, recommended values
                between 0.1 and 0.001.
            n_iter (int, optional): Number of iterations to perform the fit,
                recommended value between 5 and 10.
        """

        # generating the baseline first
        spectrum_length = len(self.y_data)
        diff_matrix = sparse.csc_matrix(np.diff(np.eye(spectrum_length), 2))
        weights = np.ones(spectrum_length)
        smoothed_baseline = np.zeros(spectrum_length)
        for _ in range(n_iter):
            weight_matrix = sparse.spdiags(weights, 0, spectrum_length, spectrum_length)
            weighted_matrix = weight_matrix + lmbd * diff_matrix.dot(
                diff_matrix.transpose()
            )
            smoothed_baseline = sparse.linalg.spsolve(  # type: ignore
                weighted_matrix, weights * self.y_data
            )
            weights = asymmetric_weight * (self.y_data > smoothed_baseline) + (
                1 - asymmetric_weight
            ) * (self.y_data < smoothed_baseline)

        # updating attribute for future use
        self._baseline = smoothed_baseline

        # subtracting the baseline
        # TODO update peak coordinates if peaks were present
        self.y_data -= smoothed_baseline
        self.logger.info("Baseline corrected")

    def integrate_area(
        self, area: tuple[float, float], rule: Literal["trapz", "simps"] = "trapz"
    ) -> float:
        """Integrate the spectrum within given area

        Args:
            area (Tuple[float, float]): Tuple with left and right border (X axis
                obviously) for the desired area.
            rule (str): Method for integration, "trapz" - trapezoidal
                rule (default), "simps" - Simpson's rule.
        Returns:
            float: Definite integral within given area as approximated by given
                method.
        """

        # closest value in experimental data and its index in data array
        _, left_idx = find_nearest_value_index(self.x_data, area[0])
        _, right_idx = find_nearest_value_index(self.x_data, area[1])

        # swap if left_idx is greater than right_idx
        if left_idx > right_idx:
            self.logger.warning(
                f"Problem with integrating peak area: Left index {left_idx} is greater "
                f"than right index {right_idx}, swapping them."
            )
            left_idx, right_idx = right_idx, left_idx

        if rule == "trapz":
            return integrate.trapezoid(
                self.y_data[left_idx : right_idx + 1],
                self.x_data[left_idx : right_idx + 1],
            )

        elif rule == "simps":
            return float(
                integrate.simpson(
                    self.y_data[left_idx : right_idx + 1],
                    self.x_data[left_idx : right_idx + 1],
                )
            )

        else:
            raise ValueError(
                'Only trapezoidal "trapz" or Simpson\'s "simps" \
rules are supported!'
            )

    def integrate_peak(self, peak: float, rule: Literal["trapz", "simps"] = "trapz"):
        """Calculate an area for a given peak

        Args:
            peak (float): (rounded) peak Y coordinate. If precise peak position
                was not found, closest is picked.
            rule (str): Method for integration, "trapz" - trapezoidal
                rule (default), "simps" - Simpson's rule.
        Returns:
            float: Definite integral within given area as approximated by given
                method.
        """

        if self.peaks is None:
            self.find_peaks()

        # this second check is purely for type checking because find_peaks() will always
        # set self.peaks to an array
        if self.peaks is None:
            raise ValueError("No peaks found, check the spectrum data.")

        true_peak, idx = find_nearest_value_index(self.peaks[:, 0], peak)
        _, _, _, left, right = self.peaks[idx]

        self.logger.debug(
            "Integrating peak found at %s, borders %.02f-%.02f", true_peak, left, right
        )

        return self.integrate_area((left, right), rule=rule)

    def smooth_spectrum(self, window_length=15, polyorder=7, in_place=True):
        """Smoothes the spectrum using Savitsky-Golay filter.

        For details see scipy.signal.savgol_filter.

        Default values for window length and polynomial order were chosen
        arbitrary based on Raman spectra.

        Args:
            window_length (int): The length of the filter window (i.e. the
                number of coefficients). window_length must be a positive odd
                integer.
            polyorder (int): The order of the polynomial used to fit the
                samples. polyorder must be less than window_length.
            in_place (bool, optional): If smoothing happens in place, returns
                smoothed spectrum if True.
        """

        if in_place:
            self.y_data = signal.savgol_filter(
                self.y_data, window_length=window_length, polyorder=polyorder
            )
            return True

        return signal.savgol_filter(
            self.y_data,
            window_length=window_length,
            polyorder=polyorder,
        )

    def default_processing(self) -> tuple[NDArray, NDArray, float | None]:
        """Dummy method to return spectral data.

        Normally redefined in ancestor classes to include basic processing for
            specific spectrum type.

        Returns:
            Tuple[NDArray, NDArray, float]: Spectral data as X and Y
                coordinates and a timestamp.
        """

        return self.x_data, self.y_data, self.timestamp

    def scale(self, factor: float):
        self.y_data = self.y_data * factor
        return self

    def normalise(self):
        self.y_data = self.y_data / np.max(self.y_data)
        return self

    def remove_mean(self):
        self.y_data = self.y_data - np.mean(self.y_data)
        return self

    def scale_to_unit_variance(self):
        self.y_data = self.y_data / np.std(self.y_data)
        return self
