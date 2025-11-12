"""
Comprehensive validation tests for NLSE solver components
"""

import numpy as np
from fourier import fft, ifft
from utils import compute_energy_time
from pulses import create_pulse

def test_fft_against_numpy():
    print("Testing FFT implementation...")
    rng = np.random.default_rng(1234)
    sizes = [1, 2, 4, 8, 16, 31, 32, 63, 64, 100, 127, 128, 256]
    
    for n in sizes:
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        X_manual = fft(x)
        X_np = np.fft.fft(x)
        
        if not np.allclose(X_manual, X_np, rtol=1e-6, atol=1e-9):
            print(f"  FAILED: FFT mismatch for n={n}")
            print(f"    Max error: {np.max(np.abs(X_manual - X_np))}")
            return False
            
        x_back = ifft(X_manual)
        if not np.allclose(x_back, x, rtol=1e-6, atol=1e-9):
            print(f"  FAILED: IFFT roundtrip mismatch for n={n}")
            print(f"    Max error: {np.max(np.abs(x_back - x))}")
            return False
    
    print("  PASSED: FFT tests for all sizes")
    return True

def test_nonlinear_energy_preserved():
    print("Testing nonlinear energy conservation...")
    from nonlinear_solver import nonlinear_phase_step
    
    t = np.linspace(-5.0, 5.0, 512)
    A0 = np.exp(-0.5 * (t / 0.5) ** 2).astype(np.complex128)
    gamma = 1.0
    alpha = 0.0  # No loss
    dz = 0.01
    
    A1 = nonlinear_phase_step(A0.copy(), gamma=gamma, alpha=alpha, dz=dz)
    E0 = compute_energy_time(A0, t)
    E1 = compute_energy_time(A1, t)
    
    if not np.allclose(E0, E1, rtol=1e-7, atol=1e-9):
        print(f"  FAILED: Nonlinear step changed energy (E0={E0:.6e}, E1={E1:.6e})")
        return False
    
    print("  PASSED: Nonlinear energy conservation")
    return True

def test_linear_dispersion():
    print("Testing linear dispersion...")
    from linear_solver import linear_propagate
    
    # Gaussian pulse parameters
    t0 = 1.0
    t = np.linspace(-20.0, 20.0, 1024)
    A0 = np.exp(-0.5 * (t / t0) ** 2).astype(np.complex128)
    
    # Propagation parameters
    beta2 = -20.0  # ps^2/km
    alpha = 0.0
    z = 0.05  # km
    
    # Numerical solution
    fields, z_arr = linear_propagate(A0, t, beta2, alpha, z, n_steps=100)
    A_num = fields[-1]
    
    # Analytical solution for Gaussian in dispersive medium
    # A(z,t) = A0 * exp(-t^2 / (2*(t0^2 + i*beta2*z))) / sqrt(1 + i*beta2*z/t0^2)
    q = 1.0 + 1j * beta2 * z / (t0 ** 2)
    A_analytical = np.exp(-0.5 * (t / t0) ** 2 / q) / np.sqrt(q)
    
    # Compare intensity profiles (phase can differ by global constant)
    I_num = np.abs(A_num) ** 2
    I_ana = np.abs(A_analytical) ** 2
    
    # Normalize for comparison
    I_num /= np.max(I_num)
    I_ana /= np.max(I_ana)
    
    max_error = np.max(np.abs(I_num - I_ana))
    if max_error > 1e-3:
        print(f"  FAILED: Linear dispersion error too large: {max_error:.6e}")
        return False
    
    print(f"  PASSED: Linear dispersion (max error: {max_error:.6e})")
    return True

def test_pulse_creation():
    print("Testing pulse creation...")
    
    t = np.linspace(-10, 10, 512)
    
    # Test Gaussian
    A_gauss = create_pulse(t, "gaussian", t0=1.0, peak_power=1.0, norm_type="unit")
    if not np.isclose(np.max(np.abs(A_gauss)**2), 1.0, rtol=1e-6):
        print("  FAILED: Gaussian unit normalization")
        return False
    
    # Test sech
    A_sech = create_pulse(t, "sech", t0=1.0, peak_power=1.0, norm_type="unit")
    if not np.isclose(np.max(np.abs(A_sech)**2), 1.0, rtol=1e-6):
        print("  FAILED: Sech unit normalization")
        return False
    
    # Test physical normalization
    P0 = 50.0
    A_phys = create_pulse(t, "gaussian", t0=1.0, peak_power=P0, norm_type="physical")
    if not np.isclose(np.max(np.abs(A_phys)**2), P0, rtol=1e-6):
        print(f"  FAILED: Physical normalization (expected {P0}, got {np.max(np.abs(A_phys)**2)})")
        return False
    
    print("  PASSED: Pulse creation")
    return True

def test_ssfm_energy_conservation():
    print("Testing SSFM energy conservation...")
    from nlse_solver import propagate_nlse
    
    t = np.linspace(-20.0, 20.0, 512)
    A0 = create_pulse(t, "gaussian", t0=1.0, peak_power=10.0, norm_type="physical")
    
    beta2 = -20.0
    gamma = 1.0
    alpha = 0.0  # No loss
    z_max = 0.1
    n_steps = 200
    
    fields, z = propagate_nlse(A0, t, beta2, gamma, alpha, z_max, n_steps)
    
    E0 = compute_energy_time(fields[0], t)
    Ef = compute_energy_time(fields[-1], t)
    
    rel_error = abs(Ef - E0) / E0
    if rel_error > 1e-4:  # Allow small numerical error
        print(f"  FAILED: SSFM energy not conserved (rel error: {rel_error:.6e})")
        return False
    
    print(f"  PASSED: SSFM energy conservation (rel error: {rel_error:.6e})")
    return True

def run_all_tests():
    print("="*60)
    print("Running NLSE Solver Test Suite")
    print("="*60)
    
    tests = [
        test_fft_against_numpy,
        test_pulse_creation,
        test_nonlinear_energy_preserved,
        test_linear_dispersion,
        test_ssfm_energy_conservation,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(False)
        print()
    
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Test Results: {passed}/{total} passed")
    print("="*60)
    
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)