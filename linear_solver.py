"""
Implements the linear step of the Split-Step Fourier Method (SSFM),
using the FFT to apply the dispersive phase shift in the frequency domain.

Equation solved:
    ∂A/∂z = i(β₂/2) ∂²A/∂t²
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from parameters import (
    T_MAX, N_T, BETA2, Z_MAX, NZ, DZ,
    INITIAL_PULSE, PULSE_T0, PLOT_DIR
)

# Time grid (ps)
dt = (2 * T_MAX) / N_T
t = np.linspace(-T_MAX, T_MAX - dt, N_T)

# Frequency grid (rad/ps)
freq = np.fft.fftfreq(N_T, d=dt)
omega = 2 * np.pi * freq

#Initial Pulse:
def initial_field(t, kind="sech", t0=1.0):
    if kind.lower() == "sech":
        return 1 / np.cosh(t / t0)
    elif kind.lower() == "gaussian":
        return np.exp(-t**2 / (2 * t0**2))
    else:
        raise ValueError("Unknown pulse shape.")

A0 = initial_field(t, INITIAL_PULSE, PULSE_T0)
A0 = A0 / np.max(np.abs(A0))  # normalize to 1

# Linear Propagator:
# For the dispersive term: A(z+dz, ω) = A(z, ω) * exp(i β₂ ω² dz / 2)
def linear_operator(omega, beta2, dz):
    return np.exp(1j * 0.5 * beta2 * omega**2 * dz)

# Propagation:
A_t = A0.copy()
z_points = np.linspace(0, Z_MAX, NZ)
fields = []

print(f"Linear propagation: {NZ} steps over {Z_MAX} km (dispersion only)...")

for z in z_points:
    fields.append((z, A_t.copy()))

    # Linear step in frequency domain (using FFT):
    A_w = np.fft.fft(A_t)
    A_w *= linear_operator(omega, BETA2, DZ)
    A_t = np.fft.ifft(A_w)

print("Linear propagation complete.\n")

# Energy conservation check (should be exact for linear case)
energies = [np.sum(np.abs(A)**2) * dt for _, A in fields]
energy_variation = np.std(energies) / np.mean(energies) * 100
print(f"Energy variation: {energy_variation:.4f}% (should be ~0 for linear case)")

# Pulse width evolution (track dispersive broadening)
def pulse_width_rms(t, A):
    # Calculate RMS pulse width
    I = np.abs(A)**2
    I_norm = I / np.sum(I)
    t_mean = np.sum(t * I_norm)
    t_rms = np.sqrt(np.sum((t - t_mean)**2 * I_norm))
    return t_rms

widths = [pulse_width_rms(t, A) for _, A in fields]
print(f"Initial width: {widths[0]:.3f} ps")
print(f"Final width:   {widths[-1]:.3f} ps")
print(f"Broadening factor: {widths[-1]/widths[0]:.2f}x\n")

# Plotting:
os.makedirs(PLOT_DIR, exist_ok=True)

# Temporal profiles
plt.figure(figsize=(8, 5))
step = max(1, NZ // 6)
for i in range(0, NZ, step):
    z, A = fields[i]
    plt.plot(t, np.abs(A)**2, label=f"z={z:.3f} km")

plt.xlim(-10*PULSE_T0, 10*PULSE_T0)
plt.xlabel("Time (ps)")
plt.ylabel("|A(t)|²")
plt.title("Pulse Evolution under Dispersion Only")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "linear_solver_temporal.png"), dpi=200)
plt.close()

# Spectrum evolution (should be unchanged in linear case)
plt.figure(figsize=(8, 5))
freq_THz = np.fft.fftshift(freq)
for i in range(0, NZ, step):
    z, A = fields[i]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(A)))**2
    plt.plot(freq_THz, spectrum / np.max(spectrum), label=f"z={z:.3f} km")

plt.xlim(-5.0 / PULSE_T0, 5.0 / PULSE_T0)
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized spectral power")
plt.title("Spectral Evolution — Dispersion Only")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "linear_solver_spectral.png"), dpi=200)
plt.close()

# Pulse width evolution
plt.figure(figsize=(8, 5))
plt.plot(z_points, widths, 'b-', linewidth=2)
plt.xlabel("Propagation distance z (km)")
plt.ylabel("RMS pulse width (ps)")
plt.title("Dispersive Broadening")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "linear_solver_width.png"), dpi=200)
plt.close()

# Save data
np.savez(os.path.join(PLOT_DIR, "fields_linear.npz"),
         t=t, freq=freq_THz, z_points=z_points,
         fields_t=np.array([f[1] for f in fields], dtype=complex),
         widths=np.array(widths))

print("Linear solver completed. Results saved in:", PLOT_DIR)