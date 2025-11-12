"""
Nonlinear propagation without dispersion
Solves: i dA/dz + gamma|A|^2A - i*(alpha/2)*A = 0
Units:
z in km, gamma in 1/(W*km), alpha in 1/km
"""

import numpy as np
import logging
from parameters import (
    GAMMA, ALPHA, Z_MAX, NZ, T_MAX, N_T,
    INITIAL_PULSE, PULSE_T0, PULSE_PEAK_POWER, NORM_TYPE
)
from pulses import create_pulse
from utils import (
    compute_energy_time,
    energy_conservation_report,
    save_results_npz,
)

# Configure logging for this module
logger = logging.getLogger(__name__)

def nonlinear_phase_step(A: np.ndarray, gamma: float, alpha: float, dz: float) -> np.ndarray:
    return A * np.exp(1j * gamma * np.abs(A) ** 2 * dz - 0.5 * alpha * dz)

def nonlinear_propagate(
    A0: np.ndarray,
    t: np.ndarray,
    gamma: float,
    alpha: float,
    z_max: float,
    n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    dz = z_max / n_steps
    A = A0.copy()
    z_values = np.linspace(0, z_max, n_steps + 1)
    fields = [A.copy()]

    for i in range(n_steps):
        A = nonlinear_phase_step(A, gamma, alpha, dz)
        fields.append(A.copy())
        if (i + 1) % max(1, n_steps // 10) == 0:
            logger.info(f"Nonlinear step {i+1}/{n_steps}")

    return np.array(fields), z_values

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    t = np.linspace(-T_MAX, T_MAX, N_T)
    A0 = create_pulse(t, INITIAL_PULSE, PULSE_T0, PULSE_PEAK_POWER, NORM_TYPE)

    logger.info("Starting nonlinear propagation...")
    logger.info(f"Pulse type: {INITIAL_PULSE}, t0={PULSE_T0} ps, P0={PULSE_PEAK_POWER} W")
    logger.info(f"gamma={GAMMA} 1/(W*km), alpha={ALPHA} 1/km")
    
    fields, z = nonlinear_propagate(A0, t, GAMMA, ALPHA, Z_MAX, NZ)
    logger.info("Nonlinear propagation complete.")

    energies = np.array([compute_energy_time(f, t) for f in fields])
    report = energy_conservation_report(energies, z, ALPHA)
    logger.info(f"Energy conservation report:")
    logger.info(f"  Initial energy: {report['E_initial']:.6e}")
    logger.info(f"  Final energy: {report['E_final']:.6e}")
    logger.info(f"  Relative change: {report['relative_change']*100:.4f}%")
    logger.info(f"  Expected loss: {report['expected_loss_percent']:.4f}%")

    save_results_npz("nonlinear_solver_results.npz", t, z, fields, save_every=10, overwrite=True)
    logger.info("Saved nonlinear solver results to nonlinear_solver_results.npz.")

if __name__ == "__main__":
    main()