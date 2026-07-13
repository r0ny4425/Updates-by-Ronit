from __future__ import annotations

"""Shared numerical constants for dense quantum-state math.

The constants in this module are intentionally small and package-local.  They
define default floating-point tolerances, the shared complex dtype, and a
precomputed square-root value reused by gate and basis constructors.
"""

import numpy as np

ATOL = 1e-12
RTOL = 1e-12
PROB_ATOL = 1e-12
SQRT2 = float(np.sqrt(2.0))
COMPLEX_DTYPE = np.complex128

__all__ = ["ATOL", "COMPLEX_DTYPE", "PROB_ATOL", "RTOL", "SQRT2"]
