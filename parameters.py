"""
Units are chosen for convenience:
- Time in picoseconds (ps)
- Length in kilometers (km)
- Wavelength in nanometers (nm)
- Dispersion (beta2) in ps^2/km
- Nonlinearity (gamma) in 1/(W·km)
"""

import numpy as np
import os

# Simulation Grid:
T_MAX = 50.0         # half-width of the time window (ps)
N_T = 2**12          # number of time samples (must be power of 2 for FFT)

# Propagation Parameters:
Z_MAX = 0.2           # total propagation length (km)
NZ = 400              # number of steps or frames to record
DZ = Z_MAX / NZ       # step size (km)

# Fiber Physical Parameters:
BETA2 = -20.0         # ps^2/km  (anomalous dispersion; negative sign)
GAMMA = 1.3           # 1/(W·km)  (nonlinear coefficient, typical for silica fiber)
ALPHA = 0.0           # loss (1/km); set nonzero if you want attenuation later

# Pulse Parameters:
INITIAL_PULSE = "sech"   # "sech" or "gaussian"
PULSE_T0 = 1.0           # width parameter (ps)
PULSE_PEAK_POWER = 50   # W (peak power of pulse)
CENTER_WAVELENGTH = 1550.0  # nm (telecom wavelength)
NORM_TYPE = "physical"   # "unit" normalizes peak to 1, "physical" keeps W units

# Derived Quantities:
C = 299792458 # Speed of light in vacuum (m/s)
CENTER_FREQUENCY = C / (CENTER_WAVELENGTH * 1e-9) / 1e12  # Convert center wavelength to frequency (THz)

# Soliton parameters (for diagnostics):
# Fundamental soliton condition: L_D = L_NL => N^2 = 1
# where L_D = t0^2 / |beta2|, L_NL = 1 / (gamma * P0)
L_D = PULSE_T0**2 / abs(BETA2)  # dispersion length (km)
L_NL = 1.0 / (GAMMA * PULSE_PEAK_POWER) if PULSE_PEAK_POWER > 0 else np.inf  # nonlinear length (km)
SOLITON_ORDER = np.sqrt(L_D / L_NL) if L_NL < np.inf else 0.0  # N parameter

# Output:
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Utility Functions:
def print_parameters():
    print("=" * 60)
    print("Simulation Parameters:")
    print(f"Time window: ±{T_MAX} ps, Samples: {N_T}")
    print(f"Propagation: Z_MAX={Z_MAX} km, Steps={NZ}, Δz={DZ:.4f} km")
    print("\nFiber Parameters")
    print(f"β₂ = {BETA2} ps²/km, γ = {GAMMA} 1/(W·km), α = {ALPHA} 1/km")
    print("\nPulse Parameters")
    print(f"Shape: {INITIAL_PULSE}, T₀ = {PULSE_T0} ps, Peak Power = {PULSE_PEAK_POWER} W")
    print(f"Central Wavelength: {CENTER_WAVELENGTH} nm")
    print("\nCharacteristic Lengths:")
    print(f"L_D (dispersion) = {L_D:.4f} km")
    print(f"L_NL (nonlinear) = {L_NL:.4f} km")
    print(f"Soliton order N = {SOLITON_ORDER:.3f}")
    if abs(SOLITON_ORDER - 1.0) < 0.15:
        print("  → Near fundamental soliton regime!")
    elif 1.5 <= SOLITON_ORDER < 2.5:
        print("  → Second-order soliton regime (expect periodic oscillations)")
    elif SOLITON_ORDER >= 2.5:
        print("  → Higher-order soliton regime (expect complex dynamics)")
    elif SOLITON_ORDER < 0.85:
        print("  → Dispersion-dominated regime")
    print("=" * 60)

def validate_parameters():
    dt = (2 * T_MAX) / N_T
    freq_max = 1.0 / (2 * dt)  # Nyquist frequency (THz)
    
    # Check temporal resolution
    samples_per_pulse = PULSE_T0 / dt
    if samples_per_pulse < 10:
        print(f"WARNING: Only {samples_per_pulse:.1f} samples per pulse width!")
        print("         Consider increasing N_T for better resolution.")
    
    # Check spectral coverage (rule of thumb: need ~10x pulse bandwidth)
    pulse_bandwidth = 1.0 / PULSE_T0  # approximate THz
    if freq_max < 10 * pulse_bandwidth:
        print(f"WARNING: Nyquist freq {freq_max:.2f} THz may be insufficient")
        print(f"         for pulse BW ~{pulse_bandwidth:.2f} THz with SPM broadening.")
    
    # Estimate SPM broadening and check against Nyquist
    max_phase = GAMMA * PULSE_PEAK_POWER * Z_MAX
    spm_broadening_factor = 1 + max_phase  # rough estimate
    final_bandwidth = pulse_bandwidth * spm_broadening_factor
    if final_bandwidth > freq_max / 3:
        print(f"WARNING: Estimated SPM bandwidth ~{final_bandwidth:.2f} THz")
        print(f"         approaches Nyquist limit {freq_max:.2f} THz.")
        print("         Risk of spectral aliasing! Consider increasing N_T.")
    
    # Check step size for dispersion (CFL-like condition)
    # Rule: dz << L_D to resolve dispersive evolution
    if DZ > 0.1 * L_D:
        print(f"WARNING: Step size Δz={DZ:.4f} km is large compared to L_D={L_D:.4f} km")
        print("         Consider decreasing DZ for better accuracy.")
    
    # Check step size for nonlinearity
    max_phase_shift = GAMMA * PULSE_PEAK_POWER * DZ
    if max_phase_shift > 0.5:  # more than ~0.5 rad per step can cause issues
        print(f"WARNING: Max nonlinear phase per step = {max_phase_shift:.3f} rad")
        print("         This may be too large; consider smaller DZ.")
    
    print("Parameter validation complete.\n")


if __name__ == "__main__":
    print_parameters()
    validate_parameters()