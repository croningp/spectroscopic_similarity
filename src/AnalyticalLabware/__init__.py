# physical instruments
from .analysis.spinsolve_spectrum import SpinsolveNMRSpectrum
from .devices.Magritek.Spinsolve.spinsolve import SpinsolveNMR
from .devices.OceanOptics.IR.NIRQuest512 import NIRQuest512
from .devices.OceanOptics.Raman.raman_control import OceanOpticsRaman

# classes for spectra processing
from .devices.OceanOptics.Raman.raman_spectrum import RamanSpectrum
from .devices.OceanOptics.UV.QEPro2192 import QEPro2192

# simulated instruments
from .devices.simulated_devices import (
    SimOceanOpticsRaman,
    _SimulatedNMRSpectrum,
    _SimulatedSpectrum,
)
