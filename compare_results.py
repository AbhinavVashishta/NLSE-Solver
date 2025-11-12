"""
compare_results.py - Compare output of linear, nonlinear, and full NLSE solvers.
Reads results from .npz files and plots temporal and spectral evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from fourier import fft, fftshift, fftfreq

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Utilities
def load_results(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing data file: {filename}")
    data = np.load(filename)
    return data["t"], data["z"], data["fields"]

def compute_spectrum(A, t):
    dt = t[1] - t[0]
    freqs = fftfreq(len(t), d=dt)
    omega = 2 * np.pi * freqs
    S = np.abs(fftshift(fft(A))) ** 2
    # Normalize by total spectral energy (prevents invisible linear case)
    S /= np.trapz(S, omega)
    return omega, S

def compute_energy(fields, t):
    dt = t[1] - t[0]
    return np.sum(np.abs(fields) ** 2, axis=1) * dt


# Main comparison function
def main():
    try:
        t_lin, z_lin, fields_lin = load_results("linear_solver_results.npz")
        t_non, z_non, fields_non = load_results("nonlinear_solver_results.npz")
        t_full, z_full, fields_full = load_results("nlse_solver_results.npz")
    except FileNotFoundError as e:
        print("Error: Missing data file. Run all solvers first.")
        print(f"  {e}")
        raise SystemExit

    # Final temporal intensity comparison
    A_lin_final = fields_lin[-1]
    A_non_final = fields_non[-1]
    A_full_final = fields_full[-1]

    plt.figure(figsize=(8, 5))
    plt.plot(t_lin, np.abs(A_lin_final) ** 2, label="Linear", lw=1.2)
    plt.plot(t_non, np.abs(A_non_final) ** 2, label="Nonlinear", lw=1.2)
    plt.plot(t_full, np.abs(A_full_final) ** 2, label="Full NLSE", lw=1.2)
    plt.xlabel("Time (ps)")
    plt.ylabel("Intensity |A|² (arb. units)")
    plt.title("Final temporal intensity comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "compare_temporal_intensity.png"), dpi=200)
    plt.close()

    # Final spectral comparison
    omega_lin, S_lin = compute_spectrum(A_lin_final, t_lin)
    omega_non, S_non = compute_spectrum(A_non_final, t_non)
    omega_full, S_full = compute_spectrum(A_full_final, t_full)

    plt.figure(figsize=(8, 5))
    plt.semilogy(omega_lin, S_lin + 1e-16, label="Linear", lw=1.2)
    plt.semilogy(omega_non, S_non + 1e-16, label="Nonlinear", lw=1.2)
    plt.semilogy(omega_full, S_full + 1e-16, label="Full NLSE", lw=1.2)
    plt.xlabel("Angular frequency (rad/ps)")
    plt.ylabel("Normalized spectral power (log scale)")
    plt.title("Final spectral intensity comparison")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "compare_spectra.png"), dpi=200)
    plt.close()

    # Energy evolution
    E_lin = compute_energy(fields_lin, t_lin)
    E_non = compute_energy(fields_non, t_non)
    E_full = compute_energy(fields_full, t_full)

    plt.figure(figsize=(8, 5))
    plt.plot(z_lin, E_lin, label="Linear")
    plt.plot(z_non, E_non, label="Nonlinear")
    plt.plot(z_full, E_full, label="Full NLSE")
    plt.xlabel("Propagation distance z (km)")
    plt.ylabel("Energy (arb. units)")
    plt.title("Energy conservation check")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "compare_energy.png"), dpi=200)
    plt.close()

    # Report
    print("\nComparison complete. Plots saved to 'plots/' folder:")
    print(" - compare_temporal_intensity.png")
    print(" - compare_spectra.png")
    print(" - compare_energy.png")
    
if __name__ == "__main__":
    main()
