"""
fourier.py - FFT utilities with a Bluestein implementation for arbitrary N.

Provides:
 - fft(x): forward DFT
 - ifft(X): inverse DFT
 - fftshift, ifftshift, fftfreq
 - internal radix-2 FFT used as building block
"""

from typing import Union
import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]

def _to_complex_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.complex128)

def _next_power_of_two(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()

def _bit_reversed_indices(n: int) -> np.ndarray:
    bits = int(np.log2(n))
    rev = np.zeros(n, dtype=int)
    for i in range(n):
        b = 0
        x = i
        for _ in range(bits):
            b = (b << 1) | (x & 1)
            x >>= 1
        rev[i] = b
    return rev


def _radix2_fft(a: np.ndarray) -> np.ndarray:
    #Iterative in-place radix-2 FFT (returns new array
    n = a.shape[0]
    if n == 1:
        return a.astype(np.complex128)
    if n & (n - 1):
        raise ValueError("radix-2 FFT requires n to be power of two")

    A = a.astype(np.complex128, copy=True)
    rev = _bit_reversed_indices(n)
    A = A[rev]

    m = 2
    while m <= n:
        half = m // 2
        wm = np.exp(-2j * np.pi / m)
        for k in range(0, n, m):
            w = 1.0 + 0j
            for j in range(half):
                t = w * A[k + j + half]
                u = A[k + j]
                A[k + j] = u + t
                A[k + j + half] = u - t
                w *= wm
        m *= 2
    return A


def _radix2_ifft(X: np.ndarray) -> np.ndarray:
    #Inverse radix-2 IFFT using conjugation trick and 1/N normalization
    n = len(X)
    if n == 0:
        return np.array([], dtype=np.complex128)
    x_conj = _radix2_fft(np.conjugate(X))
    return np.conjugate(x_conj) / n


def _bluestein_fft(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if n == 0:
        return np.array([], dtype=np.complex128)
    if n == 1:
        return x.astype(np.complex128)

    x = x.astype(np.complex128, copy=True)
    k = np.arange(n)
    # chirp: w[k] = exp(-i*pi*k^2 / n)
    w = np.exp(-1j * np.pi * (k ** 2) / n)

    a = x * w  # length n
    m = 2 * n - 1
    m_fft = _next_power_of_two(m)

    # build b[0..m_fft-1] such that circular conv of a_padded with b equals required conv
    b = np.zeros(m_fft, dtype=np.complex128)
    # b[j] = exp(i*pi*j^2 / n) for j = 0..n-1
    b[:n] = np.exp(1j * np.pi * (k ** 2) / n)
    # remaining indices correspond to negative indices: b[m_fft - (n-1) .. m_fft-1] = b[-(n-1)..-1]
    if n > 1:
        b[m_fft - (n - 1):] = np.exp(1j * np.pi * ((k[1:])[::-1] ** 2) / n)

    # pad a to length m_fft
    a_padded = np.zeros(m_fft, dtype=np.complex128)
    a_padded[:n] = a

    # compute convolution via radix-2 FFT (power-of-two length guaranteed)
    A = _radix2_fft(a_padded)
    B = _radix2_fft(b)
    C = A * B
    c = _radix2_ifft(C)

    # result: X[k] = w[k] * c[k]
    result = c[:n] * w
    return result


def fft(x: ArrayLike) -> np.ndarray:
    #Computes DFT of x. Uses radix-2 FFT when possible, Bluestein otherwise
    a = _to_complex_array(x)
    n = a.shape[0]
    if n == 0:
        return np.array([], dtype=np.complex128)
    if (n & (n - 1)) == 0:
        return _radix2_fft(a)
    return _bluestein_fft(a)


def ifft(X: ArrayLike) -> np.ndarray:
    #Computes inverse DFT. Uses conjugation trick and the forward FFT above
    A = _to_complex_array(X)
    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=np.complex128)
    x = np.conjugate(fft(np.conjugate(A)))
    return x / n


def fftshift(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x)
    return np.roll(a, a.shape[0] // 2)


def ifftshift(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x)
    return np.roll(a, -a.shape[0] // 2)


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    k = np.arange(n)
    half = n // 2
    freqs = np.where(k <= half, k, k - n) / (d * n)
    return freqs.astype(float)
