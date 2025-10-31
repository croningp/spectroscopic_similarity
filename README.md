# Utilizing Similarity Measures to Map Chemical Reactivity

The development of fully autonomous chemical synthesis platforms requires robust, real-time assessment of reactivity that does not rely on prior mechanistic knowledge. Existing methods often depend on predefined reaction models or chemical intuition, limiting their generalizability and adaptability. To address this challenge, we introduce a chemically agnostic approach that quantifies reactivity dynamics by applying similarity metrics to the full informational content of in-situ spectroscopic data. Using NMR, UV/Vis, IR, and EPR spectroscopy, we demonstrate that spectral similarity trajectories can reliably indicate reaction progress, detect kinetic features such as autocatalysis, and resolve complex behaviours including oscillations. Across diverse reaction classes, this method enabled estimated end-point detection and kinetic profiling without reaction-specific tuning. For example, the formation of lophine was followed by NMR at different reaction temperatures revealing Arrhenius-type kinetics. In a different experiment, the Belousov Zhabotinsky reaction was monitored by UV/Vis and the chemical oscillation with a periodicity of 7.25 seconds was captured. These results establish a generalizable framework for real-time, data-driven reactivity monitoring, representing a critical step toward autonomous synthesis guided by multidimensional spectroscopic feedback.

# Installation

1. Clone the repo and change your working directory

```bash
git clone https://github.com/croningp/spectroscopic_similarity.git

cd spectroscopic_similarity
```

1. Create an environment and activate it

```bash
conda create -n spec_sim python=3.12
```

```bash
conda activate spec_sim
```

2. Install AnalyticalLabware

```bash
pip install -e .
```

## Requirements

### Python libraries

- scipy
- matplotlib
- numpy
- nmrglue (python library for nmr data processing, [git][nmrglue-git]/[docs][nmrglue-docs])

# Usage guides

## Spinsolve NMR

```python
from AnalyticalLabware.devices import SpinsolveNMR

# Make sure that Spinsolve software is running and
# the Remote Control option is turned on
s = SpinsolveNMR()

# create a data folder to save spectra to
s.user_folder("path_to_folder")  # check available saving options in method doc

# set experiment specific data
s.solvent = "methanol"
s.sample = "TEST-1"
s.user_data = {"comment": "test experiment 1"}

s.protocols_list()  # yields list of all available protocols

# get available protocol options
# where 'protocol_name' is the name of the protocol
s.cmd.get_protocol("protocol_name")

# shim on sample
# check available shimming options in the manual
s.shim_on_sample(
    reference_peak="<reference_peak_float_number>", option="<option_parameter>"
)

# simple proton experiment
# where 'option' is a valid option for simple proton protocol ('1D PROTON')
s.proton(option="option")

### Basic spectrum processing is available through s.spectrum class ###
# loads last measured spectrum if any, else runs default protocol
s.get_spectrum()

# now ppm and spectral data are available as s.spectrum.x and s.spectrum.y
# hint: to obtain raw FID data use the following public method for the 'data.1d' file
time_axis, fid_real, fid_imag = s.spectrum.extract_data("<path_to_fid_file>")
```

## OceanOptics Raman spectrometer

```python
from AnalyticalLabware.devices import OceanOpticsRaman

# make sure your instrument is connected and you've
# installed all required hardware drivers
r = OceanOpticsRaman()

# set the integration time in seconds!
r.set_integration_time("<time_in_seconds>")

# obtain raw spectroscopic data
# will return tuple of wavelength and intensities arrays
# as np.array objects
r.scan("<n_scans>")
```

The module also supports basic spectrum processing.

```python
# load the spectrum data to internal Spectrum class
r.get_spectrum()

# the spectrum is now available via r.spectrum
# you can use the following methods for processing
r.spectrum.correct_baseline()  # subtracting the baseline
r.spectrum.smooth_spectrum()  # smoothing the spectrum
r.spectrum.trim(xmin, xmax)  # trimming the spectrum on X axis

# finding the peaks, please refer to method documentation for
# attribute assignment
r.spectrum.find_peaks(threshold, min_width, min_dist)
# alternative method
# r.spectrum.find_peaks_iteratively(threshold, steps)

# integration
r.spectrum.integrate_area(area)
r.spectrum.integrate_peak(peak)

# saving the data as .pickle file
r.spectrum.save_data(filename)  # !filename without .pickle extension!
```

## Chemputer specific

AnalyticalLabware devices can be used in chempiler enironment if added to the corresponding graph with the correct set of parameters.

Example of the Spinsolve NMR object on the graph:

```json
{
  "id": "nmr",
  "type": "custom",
  "x": 240,
  "y": 240,
  "customProperties": {
    "name": {
      "name": "name",
      "id": "name",
      "units": "",
      "type": "str"
    }
  },
  "internalId": 24,
  "label": "nmr",
  "current_volume": 0,
  "class": "ChemputerNMR",
  "name": "nmr"
}
```

To instantiate chempiler, simply import `chemputer_devices` module from `AnalyticalLabware` and supply as a `device_modules` argument during Chempiler instantiation, e.g.

```python
from AnalyticalLabware.devices import chemputer_devices
from chempiler import Chempiler
import ChemputerAPI

c = Chempiler(
    "procedure",
    "graph.json",
    "output",
    simulation=True,
    device_modules=[ChemputerAPI, chemputer_devices],
)
```

[nmrglue-docs]: https://nmrglue.readthedocs.io/en/latest/index.html
[nmrglue-git]: https://github.com/jjhelmus/nmrglue
