"""
Sailboat building configuration, D = 6.

Material-coordinate definitions in the released deck:
  e1: Material 1 columns and roof/core shell layers
  e2: Material 2 higher-storey columns
  e3: Material 3 beam groups
  e4-e6: Materials 6-8, with no active element assignments

E_actual = e_i * 2e10 Pa; the nominal value is 1.
Forward model: ANSYS APDL, Block Lanczos, 20 modal frequencies.
See IMPLEMENTATION.md for section-layer assignments.
"""

N_PARAMS = 6
N_MODES = 20

PARAM_NAMES = ["e1", "e2", "e3", "e4", "e5", "e6"]
FREQ_NAMES = [f"f{i}" for i in range(1, N_MODES + 1)]

# Sampling bounds (uniform prior for training data coverage)
THETA_LO = 0.5
THETA_HI = 2.0
