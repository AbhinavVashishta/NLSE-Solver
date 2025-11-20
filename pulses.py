"""
pulses.py - Centralized pulse generation for NLSE solvers

Implements various initial pulse shapes with consistent normalization options.
"""

import numpy as np
from typing import Literal

def create_pulse(
    t: np.ndarray,
    pulse_type: Literal["gaussian", "sech"] = "gaussian",
    t0: float = 1.0,
    peak_power: float = 1.0,
    norm_type: Literal["physical", "unit"] = "physical"
) -> np.ndarray:
    dt = t[1] - t[0]

    if pulse_type == "gaussian":
        A = np.exp(-0.5 * (t / t0) ** 2)
    elif pulse_type == "sech":
        A = 1.0 / np.cosh(t / t0)
    else:
        raise ValueError(f"Unknown pulse_type: {pulse_type}. Use 'gaussian' or 'sech'.")

    if norm_type == "physical":
        A = A / np.max(np.abs(A))
        A = A * np.sqrt(peak_power)
    elif norm_type == "unit":
        A = A / np.max(np.abs(A))
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Use 'physical' or 'unit'.")

    return A.astype(np.complex128)


def create_chirped_pulse(
    t: np.ndarray,
    pulse_type: Literal["gaussian", "sech"] = "gaussian",
    t0: float = 1.0,
    peak_power: float = 1.0,
    chirp_C: float = 0.0,
    norm_type: Literal["physical", "unit"] = "physical"
) -> np.ndarray:
    A = create_pulse(t, pulse_type, t0, peak_power, norm_type)

    if chirp_C != 0.0:
        phase = chirp_C * (t / t0) ** 2
        A = A * np.exp(1j * phase)

    return A
