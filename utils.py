"""
Diagnostics, energy, and plotting utilities for NLSE solver
"""

import numpy as np
import os
from typing import Dict, Any
from fourier import fft, ifft, fftshift, ifftshift, fftfreq

# ENERGY CALCULATION
def compute_energy_time(A: np.ndarray, t: np.ndarray) -> float:
    if A.size == 0:
        return 0.0
    dt = float(t[1] - t[0]) if t.size > 1 else 1.0
    energy = np.sum(np.abs(A) ** 2) * dt
    return float(energy)


def compute_spectral_width_rms(A: np.ndarray, t: np.ndarray) -> float:
    n = A.size
    if n == 0:
        return 0.0
    dt = float(t[1] - t[0]) if n > 1 else 1.0
    freqs = fftfreq(n, d=dt)
    omega = 2.0 * np.pi * freqs
    A_w = fftshift(fft(A))
    S = np.abs(A_w) ** 2
    domega = omega[1] - omega[0] if n > 1 else 1.0
    omega_shifted = np.fft.fftshift(omega) 
    S_sum = np.sum(S) * domega
    if S_sum == 0:
        return 0.0
    mean_omega = np.sum(omega_shifted * S) * domega / S_sum
    var_omega = np.sum(((omega_shifted - mean_omega) ** 2) * S) * domega / S_sum
    return float(np.sqrt(var_omega))


def save_results_npz(
    filename: str,
    t: np.ndarray,
    z_array: np.ndarray,
    fields: np.ndarray,
    save_every: int = 1,
    overwrite: bool = False
) -> None:
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"{filename} already exists and overwrite=False")
    arr = np.asarray(fields)
    # Ensure fields are 2D: (n_steps+1, n_t)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], arr.shape[1])
    if arr.ndim == 2 and save_every > 1:
        arr = arr[::save_every, :]
    np.savez_compressed(
        filename,
        t=t,
        z=z_array[::save_every] if save_every > 1 else z_array,
        fields=arr
    )


def autoscale_time_axis(ax, t: np.ndarray, margin: float = 0.05) -> None:
    tmin, tmax = np.min(t), np.max(t)
    spread = (tmax - tmin)
    if spread == 0:
        ax.set_xlim(tmin - 1.0, tmax + 1.0)
    else:
        ax.set_xlim(tmin - margin * spread, tmax + margin * spread)


def autoscale_freq_axis(ax, freqs: np.ndarray, margin: float = 0.05) -> None:
    fmin, fmax = np.min(freqs), np.max(freqs)
    spread = (fmax - fmin)
    if spread == 0:
        ax.set_xlim(fmin - 1.0, fmax + 1.0)
    else:
        ax.set_xlim(fmin - margin * spread, fmax + margin * spread)


def energy_conservation_report(
    energies: np.ndarray,
    z_array: np.ndarray,
    alpha: float = 0.0
) -> Dict[str, Any]:
    E_initial = float(energies[0])
    E_final = float(energies[-1])
    rel_change = (E_final - E_initial) / E_initial if E_initial != 0 else 0.0
    expected_loss = (1.0 - np.exp(-alpha * (z_array[-1] - z_array[0]))) * 100.0 if alpha > 0 else 0.0
    return {
        "E_initial": E_initial,
        "E_final": E_final,
        "relative_change": rel_change,
        "expected_loss_percent": expected_loss
    }
