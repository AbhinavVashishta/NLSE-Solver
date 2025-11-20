"""
Generates for each solver:
  - Temporal and spectral profiles at representative z positions
  - Energy vs. propagation distance
  - Peak power vs. propagation distance
  - (Optional for nonlinear) Chirp vs. time at final z

Outputs are saved in 'plots/'.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Literal
from fourier import fft, fftshift, fftfreq
from utils import compute_energy_time

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Indices for plotting: start, middle, end
PROFILE_INDICES_START = 0
PROFILE_INDICES_MIDDLE = 0.5  # Fraction of total length
PROFILE_INDICES_END = -1

def load_results(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fname = f"{name}_solver_results.npz"
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Missing file: {fname}. Please run {name}_solver.py first."
        )
    data = np.load(fname)
    return data["t"], data["z"], data["fields"]

def plot_temporal_profiles(
    fields: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    label: str,
    fname: str
) -> None:
    mid_idx = len(z) // 2
    indices = [PROFILE_INDICES_START, mid_idx, PROFILE_INDICES_END]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(8, 5))
    for i, c in zip(indices, colors):
        plt.plot(t, np.abs(fields[i])**2, color=c, lw=1.2,
                 label=f"z = {z[i]:.3f} km")
    plt.xlabel("Time (ps)")
    plt.ylabel("|A|²")
    plt.title(f"{label}: Temporal evolution at selected z")
    plt.legend()
    plt.grid(True, alpha=0.3)
    #plt.xlim(-60,60)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=200)
    plt.close()

def plot_spectral_profiles(
    fields: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    label: str,
    fname: str
) -> None:
    n = len(t)
    dt = t[1] - t[0]
    freqs = fftshift(fftfreq(n, d=dt))
    
    mid_idx = len(z) // 2
    indices = [PROFILE_INDICES_START, mid_idx, PROFILE_INDICES_END]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(8, 5))
    for i, c in zip(indices, colors):
        A_w = np.abs(fftshift(fft(fields[i])))**2
        A_w /= np.max(A_w)
        plt.plot(freqs, A_w, color=c, lw=1.2,
                 label=f"z = {z[i]:.3f} km")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Normalized |Ã|²")
    plt.title(f"{label}: Spectral evolution at selected z")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.xlim(-0.5, 0.5)
    #plt.ylim(bottom=np.max(A_w)*1e-3)
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=200)
    plt.close()

def plot_energy_and_peak(
    fields: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    label: str,
    fname: str
) -> None:
    energies = np.array([compute_energy_time(f, t) for f in fields])
    peak_power = np.array([np.max(np.abs(f)**2) for f in fields])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(z, energies, "tab:blue", lw=1.2, label="Energy")
    ax2.plot(z, peak_power, "tab:red", lw=1.2, label="Peak |A|²")

    ax1.set_xlabel("Propagation distance z (km)")
    ax1.set_ylabel("Energy (arb. units)", color="tab:blue")
    ax2.set_ylabel("Peak |A|²", color="tab:red")
    ax1.set_title(f"{label}: Energy and Peak Power vs z")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=200)
    plt.close()

def plot_chirp(
    fields: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    label: str,
    fname: str
) -> None:
    A_final = fields[PROFILE_INDICES_END]
    phase = np.unwrap(np.angle(A_final))
    chirp = np.gradient(phase, t)
    
    plt.figure(figsize=(8, 4))
    plt.plot(t, chirp, lw=1.2, color='tab:purple')
    plt.xlabel("Time (ps)")
    plt.ylabel("dφ/dt (rad/ps)")
    plt.title(f"{label}: Instantaneous frequency (chirp) at z={z[-1]:.3f} km")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.ylim(-0.2,0.2)
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=200)
    plt.close()

def visualize_solver(
    name: Literal["linear", "nonlinear", "nlse"],
    label: str,
    include_chirp: bool = False
) -> None:
    print(f"Processing {label} solver...")
    try:
        t, z, fields = load_results(name)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print(f"  Skipping {label} visualization.")
        return

    plot_temporal_profiles(fields, t, z, label, f"{name}_temporal_evolution.png")
    plot_spectral_profiles(fields, t, z, label, f"{name}_spectral_evolution.png")
    plot_energy_and_peak(fields, t, z, label, f"{name}_energy_peak.png")

    if include_chirp:
        plot_chirp(fields, t, z, label, f"{name}_chirp.png")

    print(f"  -> Saved {label} plots to {PLOT_DIR}/")

def main():
    solvers = [
        ("linear", "Linear", False),
        ("nonlinear", "Nonlinear", True),
        ("nlse", "Full NLSE", True)
    ]
    
    success_count = 0
    for name, label, chirp in solvers:
        try:
            visualize_solver(name, label, include_chirp=chirp)
            success_count += 1
        except Exception as e:
            print(f"  Error visualizing {label}: {e}")
    
    if success_count > 0:
        print(f"\nVisualization complete. {success_count}/{len(solvers)} solvers processed.")
        print(f"Check the '{PLOT_DIR}/' folder for output.")
    else:
        print("\nNo visualizations generated. Run the solvers first.")

if __name__ == "__main__":
    main()