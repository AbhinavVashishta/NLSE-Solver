"""
Equation:
    ∂A/∂z = i (β2 / 2) ∂^2 A / ∂t^2 + i γ |A|^2 A - (α/2) A

Algorithm (per step dz):
    1) Apply half linear step:  A -> exp(i β2 ω^2 dz/4) · FFT(A) -> IFFT
    2) Apply full nonlinear step: A -> A · exp(i γ |A|^2 dz) · exp(-α dz/2)
    3) Apply half linear step again
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from parameters import (
    T_MAX, N_T, Z_MAX, NZ, DZ,
    INITIAL_PULSE, PULSE_T0, PULSE_PEAK_POWER,
    BETA2, GAMMA, ALPHA, NORM_TYPE, PLOT_DIR,
    SOLITON_ORDER, L_D, L_NL, print_parameters
)

print_parameters()

# grid and transforms:
dt = (2 * T_MAX) / N_T
t = np.linspace(-T_MAX, T_MAX - dt, N_T)            # time grid (ps)
freq = np.fft.fftfreq(N_T, d=dt)                    # cycles/ps (THz)
omega = 2.0 * np.pi * freq                          # rad/ps

# frequency axis for plotting (shifted cycles/ps -> THz)
freq_THz = np.fft.fftshift(freq)

# initial field:
def initial_field(t, kind="sech", t0=1.0):
    if kind.lower() == "sech":
        return 1.0 / np.cosh(t / t0)
    elif kind.lower() == "gaussian":
        return np.exp(-t**2 / (2 * t0**2))
    else:
        raise ValueError("Unknown pulse shape")

A_t = initial_field(t, INITIAL_PULSE, PULSE_T0)

# apply normalization option
if NORM_TYPE == "physical":
    # scale amplitude so peak power = PULSE_PEAK_POWER (A has units sqrt(W))
    A_t = A_t / np.max(np.abs(A_t)) * np.sqrt(PULSE_PEAK_POWER)
else:
    # unit peak normalization (peak = 1)
    A_t = A_t / np.max(np.abs(A_t))

# operators:
dz = DZ  # step length (km) from parameters

# Linear operator for a step of length `d` is exp(i * beta2/2 * omega^2 * d)
# For half-step we use d = dz/2
def linear_prop(omega, beta2, d):
    return np.exp(1j * 0.5 * beta2 * (omega**2) * d)

H_half = linear_prop(omega, BETA2, dz/2.0)
# We will use H_half twice each step (pre and post nonlinear)

# Nonlinear operator (time-domain): exp(i * gamma * |A|^2 * dz)
# Attenuation from alpha included as amplitude factor exp(-alpha * dz/2)
def nonlinear_phase(A, gamma, dz, alpha=0.0):
    phase = np.exp(1j * gamma * (np.abs(A)**2) * dz)
    if alpha != 0.0:
        atten = np.exp(-0.5 * alpha * dz)
        return atten * A * phase
    else:
        return A * phase

#  propagation:
z_points = np.linspace(0.0, Z_MAX, NZ)
A = A_t.copy()
fields_t = []   # (z, A_t snapshot)
fields_w = []   # (z, spectral power snapshot)

# store initial
fields_t.append((0.0, A.copy()))
fields_w.append((0.0, np.abs(np.fft.fftshift(np.fft.fft(A)))**2))

print(f"Starting propagation: {NZ} steps over {Z_MAX} km...")

for i in range(1, len(z_points)):
    z_prev = z_points[i-1]
    z_next = z_points[i]

    # half linear step (freq domain)
    A_w = np.fft.fft(A)
    A_w = A_w * H_half
    A = np.fft.ifft(A_w)

    # full nonlinear step (time domain)
    A = nonlinear_phase(A, GAMMA, dz, ALPHA)

    # half linear step again
    A_w = np.fft.fft(A)
    A_w = A_w * H_half
    A = np.fft.ifft(A_w)

    # store
    fields_t.append((z_next, A.copy()))
    fields_w.append((z_next, np.abs(np.fft.fftshift(np.fft.fft(A)))**2))

print("Propagation complete.")

os.makedirs(PLOT_DIR, exist_ok=True)

# Energy/Power check (integral of |A|^2 over time, properly normalized)
# For unit normalization, this tracks relative energy loss
# For physical units, this is actual pulse energy in W·ps
energies = [np.sum(np.abs(A_snap)**2) * dt for _, A_snap in fields_t]
energy_loss_pct = 100 * (energies[-1] - energies[0]) / energies[0]

print("\n" + "="*60)
print("Energy Conservation Check:")
print(f"  Initial: {energies[0]:.6e}")
print(f"  Mid:     {energies[len(energies)//2]:.6e}")
print(f"  Final:   {energies[-1]:.6e}")
print(f"  Loss:    {energy_loss_pct:+.3f}%")
if ALPHA == 0.0 and abs(energy_loss_pct) > 0.1:
    print("  WARNING: Energy not conserved (alpha=0)! Check numerics.")
elif ALPHA > 0.0:
    expected_loss = 100 * (np.exp(-ALPHA * Z_MAX) - 1)
    print(f"  Expected (from α): {expected_loss:+.3f}%")
print("="*60 + "\n")

# Peak power evolution (for tracking soliton behavior)
peak_powers = [np.max(np.abs(A_snap)**2) for _, A_snap in fields_t]
peak_power_change_pct = 100 * (peak_powers[-1] - peak_powers[0]) / peak_powers[0]
print(f"Peak Power Change: {peak_power_change_pct:+.2f}%")
if 0.85 <= SOLITON_ORDER <= 1.15:
    if abs(peak_power_change_pct) < 5.0:
        print("  → Stable fundamental soliton detected!")
    else:
        print("  → Near fundamental soliton with weak perturbations")
elif 1.5 <= SOLITON_ORDER < 2.5:
    print(f"  → Second-order soliton: periodic breathing (N={SOLITON_ORDER:.2f})")
elif SOLITON_ORDER >= 2.5:
    print(f"  → Higher-order soliton: complex dynamics (N={SOLITON_ORDER:.2f})")
print()

# save fields (complex) and grids
np.savez(os.path.join(PLOT_DIR, "fields_nlse.npz"),
         t=t, freq=freq_THz, z_points=z_points,
         fields_t=np.array([f[1] for f in fields_t], dtype=complex),
         fields_w=np.array([f[1] for f in fields_w], dtype=float),
         energies=np.array(energies),
         peak_powers=np.array(peak_powers))

# plotting:
# Temporal evolution snapshots (intensity)
plt.figure(figsize=(8, 5))
nplots = min(6, len(fields_t))
indices = np.linspace(0, len(fields_t)-1, nplots, dtype=int)
for idx in indices:
    z, A_snap = fields_t[idx]
    plt.plot(t, np.abs(A_snap)**2, label=f"z={z:.3f} km")
plt.xlim(-10*PULSE_T0, 10*PULSE_T0)
plt.xlabel("Time (ps)")
if NORM_TYPE == "physical":
    plt.ylabel("Intensity |A(t)|² (W)")
else:
    plt.ylabel("|A(t)|²")
plt.title("Temporal intensity — Full NLSE (SSFM)")
plt.legend(loc="upper right", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "nlse_temporal.png"), dpi=200)
plt.close()

# Spectral evolution snapshots (normalized)
plt.figure(figsize=(8, 5))
for idx in indices:
    z, A_snap = fields_t[idx]  # fixed: was accessing fields_w incorrectly
    S = np.abs(np.fft.fftshift(np.fft.fft(A_snap)))**2
    plt.plot(freq_THz, S / np.max(S), label=f"z={z:.3f} km")
plt.xlim(-5.0 / PULSE_T0, 5.0 / PULSE_T0)
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized spectral power")
plt.title("Spectral evolution — Full NLSE (SSFM)")
plt.legend(loc="upper right", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "nlse_spectral.png"), dpi=200)
plt.close()

# Energy evolution plot (diagnostic)
plt.figure(figsize=(8, 5))
plt.plot(z_points, energies, 'b-', linewidth=2)
plt.xlabel("Propagation distance z (km)")
if NORM_TYPE == "physical":
    plt.ylabel("Pulse energy (W·ps)")
else:
    plt.ylabel("Pulse energy (arb. units)")
plt.title("Energy Conservation")
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "nlse_energy.png"), dpi=200)
plt.close()

# Peak power evolution (soliton diagnostic)
plt.figure(figsize=(8, 5))
plt.plot(z_points, peak_powers, 'r-', linewidth=2)
plt.xlabel("Propagation distance z (km)")
if NORM_TYPE == "physical":
    plt.ylabel("Peak power |A|² (W)")
else:
    plt.ylabel("Peak power |A|²")
plt.title("Peak Power Evolution")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "nlse_peak_power.png"), dpi=200)
plt.close()

print("nlse_solver.py finished. Plots and data saved to:", PLOT_DIR)