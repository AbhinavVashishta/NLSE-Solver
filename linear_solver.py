"""
Linear propagation solver
Solves: i dA/dz + (beta2/2) d^2A/dt^2 - i*(alpha/2)*A = 0

"""

import numpy as np
import logging
from parameters import (
    BETA2, ALPHA, Z_MAX, NZ, T_MAX, N_T,
    INITIAL_PULSE, PULSE_T0, PULSE_PEAK_POWER, NORM_TYPE
)
from pulses import create_pulse
from fourier import fft, ifft, fftfreq
from utils import (
    compute_energy_time,
    compute_spectral_width_rms,
    energy_conservation_report,
    save_results_npz,
)

logger = logging.getLogger(__name__)


def linear_propagate(
    A0: np.ndarray,
    t: np.ndarray,
    beta2: float,
    alpha: float,
    z_max: float,
    n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    n = len(t)
    dz = z_max / n_steps
    freqs = fftfreq(n, d=dt)
    omega = 2 * np.pi * freqs

    # Linear operator
    H = np.exp(-1j * (beta2 / 2.0) * (omega ** 2) * dz - 0.5 * alpha * dz)

    A = A0.copy()
    z_values = np.linspace(0, z_max, n_steps + 1)
    fields = [A.copy()]

    for i in range(n_steps):
        A = ifft(fft(A) * H)
        fields.append(A.copy())
        if (i + 1) % max(1, n_steps // 10) == 0:
            logger.info(f"Linear step {i+1}/{n_steps}")

    return np.array(fields), z_values


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    t = np.linspace(-T_MAX, T_MAX, N_T)
    A0 = create_pulse(t, INITIAL_PULSE, PULSE_T0, PULSE_PEAK_POWER, NORM_TYPE)

    logger.info("Starting linear propagation...")
    logger.info(f"Pulse type: {INITIAL_PULSE}, t0={PULSE_T0} ps, P0={PULSE_PEAK_POWER} W")
    logger.info(f"beta2={BETA2} ps^2/km, alpha={ALPHA} 1/km")

    fields, z = linear_propagate(A0, t, BETA2, ALPHA, Z_MAX, NZ)
    logger.info("Linear propagation complete.")

    energies = np.array([compute_energy_time(f, t) for f in fields])
    report = energy_conservation_report(energies, z, ALPHA)
    logger.info(f"Energy conservation report:")
    logger.info(f"  Initial energy: {report['E_initial']:.6e}")
    logger.info(f"  Final energy: {report['E_final']:.6e}")
    logger.info(f"  Relative change: {report['relative_change']*100:.4f}%")
    logger.info(f"  Expected loss: {report['expected_loss_percent']:.4f}%")

    save_results_npz("linear_solver_results.npz", t, z, fields, save_every=10, overwrite=True)
    logger.info("Saved linear solver results to linear_solver_results.npz.")


if __name__ == "__main__":
    main()