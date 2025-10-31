import nmrglue as ng

from AnalyticalLabware.analysis.nmr_spectrum import NMRSpectrum


class BrukerNMRSpectrum(NMRSpectrum):
    """Class for Bruker NMR spectrum loading and handling.

    Exploits the existing functionalities in nmrglue."""

    def load_spectrum(self, data_path, preprocessed=False):
        # Read the Bruker directory
        dic, data = ng.bruker.read(data_path)

        # Convert to NMRPipe format (makes processing easier)
        self.udic = ng.bruker.guess_udic(dic, data)

        # changing axis mapping according to raw FID
        self.x_label = "time"

        # creating unit conversion object
        self._uc = ng.fileio.fileiobase.uc_from_udic(self.udic)

        x_axis = self._uc.sec_scale()

        super().load_data(x_axis, data, int(dic["acqus"]["DATE"]))
