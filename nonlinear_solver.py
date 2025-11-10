"""
We integrate the nonlinear term of the NLSE in time domain:
    ∂A/∂z = i γ |A|^2 A  => exact solution over a step dz:
    A(z+dz,t) = A(z,t) * exp(i γ |A|^2 dz)
    
This demonstrates pure Self-Phase Modulation (SPM) without dispersion.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from parameters import (
    T_MAX, N_T, Z_MAX, NZ, DZ,
    INITIAL_PULSE, PULSE_T0, PULSE_PEAK_POWER,
    GAMMA, ALPHA, NORM_TYPE, PLOT_DIR
)

# Time grid (ps)
dt = (2 * T_MAX) / N_T
t = np.linspace(-T_MAX, T_MAX - dt, N_T)

# Frequency grid for plotting (cycles/ps -> THz)
freq = np.fft.fftfreq(N_T, d=dt)
freq_THz = np.fft.fftshift(freq)

# Initial Pulse (time domain):
def initial_field(t, kind="sech", t0=1.0):
    if kind.lower() == "sech":
        return 1.0 / np.cosh(t / t0)
    elif kind.lower() == "gaussian":
        return np.exp(-t**2 / (2 * t0**2))
    else:
        raise ValueError("Unknown pulse shape.")

# Build initial envelope. Use physical peak power if requested.
A_t = initial_field(t, INITIAL_PULSE, PULSE_T0)

if NORM_TYPE == "physical":
    # Convert to amplitude with peak power P0: A -> sqrt(P0) * normalized_shape
    A_t = A_t / np.max(np.abs(A_t)) * np.sqrt(PULSE_PEAK_POWER)
else:
    # unit normalization (peak = 1), convenient for comparing linear/nonlinear stages
    A_t = A_t / np.max(np.abs(A_t))

# Prepare propagation
z_points = np.linspace(0, Z_MAX, NZ)
fields = []  # (z, A_t)

# Nonlinear propagator: time-domain phase-only
# include simple loss alpha as amplitude attenuation if ALPHA > 0
def nonlinear_step(A, gamma, dz, alpha=0.0):
    # alpha in 1/km - include as amplitude attenuation factor exp(-alpha*dz/2)
    if alpha != 0.0:
        attenuation = np.exp(-0.5 * alpha * dz)
    else:
        attenuation = 1.0
    phase = np.exp(1j * gamma * np.abs(A)**2 * dz)
    return attenuation * A * phase

print(f"Nonlinear-only propagation: {NZ} steps over {Z_MAX} km (SPM only)...")

# Propagate with pure nonlinear step at each dz
A = A_t.copy()
fields.append((0.0, A.copy()))  # store initial condition

for i in range(1, len(z_points)):
    # step forward by DZ using exact nonlinear solution for Kerr term
    A = nonlinear_step(A, GAMMA, DZ, ALPHA)
    fields.append((z_points[i], A.copy()))

print("Nonlinear propagation complete.\n")

# Energy check (for nonlinear-only, energy conserved if alpha=0)
energies = [np.sum(np.abs(A_snap)**2) * dt for _, A_snap in fields]
energy_loss_pct = 100 * (energies[-1] - energies[0]) / energies[0]
print(f"Energy change: {energy_loss_pct:+.3f}%")
if ALPHA == 0.0 and abs(energy_loss_pct) > 0.01:
    print("  WARNING: Energy should be conserved for SPM-only (alpha=0)!")

# Spectral broadening check
def spectral_width_rms(freq, spectrum):
    # Calculate RMS spectral width
    S_norm = spectrum / np.sum(spectrum)
    f_mean = np.sum(freq * S_norm)
    f_rms = np.sqrt(np.sum((freq - f_mean)**2 * S_norm))
    return f_rms

spectra = [np.abs(np.fft.fft(A_snap))**2 for _, A_snap in fields]
spec_widths = [spectral_width_rms(freq, S) for S in spectra]
print(f"Initial spectral width: {spec_widths[0]:.4f} THz")
print(f"Final spectral width:   {spec_widths[-1]:.4f} THz")
print(f"SPM broadening factor: {spec_widths[-1]/spec_widths[0]:.2f}x\n")

# Ensure output directory
os.makedirs(PLOT_DIR, exist_ok=True)

# Plot temporal evolution (intensity):
plt.figure(figsize=(8, 5))
step = max(1, NZ // 6)
for i in range(0, NZ, step):
    z, A_snap = fields[i]
    plt.plot(t, np.abs(A_snap)**2, label=f"z={z:.3f} km")
plt.xlim(-10*PULSE_T0, 10*PULSE_T0)
plt.xlabel("Time (ps)")
if NORM_TYPE == "physical":
    plt.ylabel("Intensity |A(t)|² (W)")
else:
    plt.ylabel("|A(t)|²")
plt.title("Temporal intensity — Nonlinear only (SPM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "nonlinear_temporal.png"), dpi=200)
plt.close()

# Plot spectral evolution (showing SPM broadening):
plt.figure(figsize=(8, 5))
for i in range(0, NZ, step):
    z, A_snap = fields[i]
    S = np.abs(np.fft.fftshift(np.fft.fft(A_snap)))**2
    plt.plot(freq_THz, S / np.max(S), label=f"z={z:.3f} km")
plt.xlim(-5.0 / PULSE_T0, 5.0 / PULSE_T0)
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized spectral power")
plt.title("Spectral evolution — Nonlinear only (SPM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "nonlinear_spectral.png"), dpi=200)
plt.close()

# Spectral width evolution
plt.figure(figsize=(8, 5))
plt.plot(z_points, spec_widths, 'r-', linewidth=2)
plt.xlabel("Propagation distance z (km)")
plt.ylabel("RMS spectral width (THz)")
plt.title("SPM Spectral Broadening")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "nonlinear_spectral_width.png"), dpi=200)
plt.close()

# Instantaneous frequency (chirp visualization)
# chirp = -d(phase)/dt for the pulse
z_final, A_final = fields[-1]
phase = np.unwrap(np.angle(A_final))
chirp = -np.gradient(phase, t)

plt.figure(figsize=(8, 5))
plt.plot(t, chirp, 'g-', linewidth=2)
plt.xlim(-10*PULSE_T0, 10*PULSE_T0)
plt.xlabel("Time (ps)")
plt.ylabel("Instantaneous frequency (rad/ps)")
plt.title(f"SPM-induced chirp at z={z_final:.3f} km")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "nonlinear_chirp.png"), dpi=200)
plt.close()

# Save numerical data for later analysis:
np.savez(os.path.join(PLOT_DIR, "fields_nonlinear.npz"),
         t=t, freq=freq_THz, z_points=z_points,
         fields_t=np.array([f[1] for f in fields], dtype=complex),
         spec_widths=np.array(spec_widths))

print("nonlinear_solver.py finished. Plots saved in", PLOT_DIR)