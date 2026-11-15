"""
Coupled-building configuration, D = 3 interface.

Parameters enter ANSYS as E_actual = scaling * 1e10 Pa:
  e1: Material 1 column and shell-layer moduli
  e2: Nominal material 2 interface coordinate
  e3: Material 3 beam moduli

Both shell sections select material 1 in their layer data. The e2
coordinate is not interpreted as independently identified shell stiffness.
Legacy CSV column names remain unchanged for compatibility.
Forward model: ANSYS APDL, Block Lanczos, 20 modal frequencies.
Measured first-six frequencies: Hu et al. (2018), Table 2.
"""

N_PARAMS = 3
N_MODES = 20

PARAM_NAMES = ["e1_column", "e2_roof", "e3_beam"]
APDL_VAR = ["e1", "e2", "e3"]

E_BASE = 1e10  # Pa; E_actual = scaling * E_BASE

# Uniform prior bounds on scaling factor
THETA_LO = 0.5
THETA_HI = 4.0
PARAM_RANGE = [THETA_LO, THETA_HI]

FREQ_NAMES = [f"f{i}" for i in range(1, N_MODES + 1)]

# Real measured modal frequencies (Hu et al. 2018 SCHM, Table 2), modes 1-6
MEASURED_FREQS = [2.06, 2.30, 2.62, 6.14, 6.53, 6.76]  # Hz
N_MEASURED = len(MEASURED_FREQS)
