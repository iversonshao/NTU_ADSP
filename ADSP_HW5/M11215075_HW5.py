import numpy as np
import matplotlib.pyplot as plt

def fftreal(f1, f2):
    N = len(f1)
    if N != len(f2):
        raise ValueError("Both sequences must have the same length")

    # Combine the two real sequences into one complex sequence
    f3 = f1 + 1j * f2
    # Perform FFT on the combined complex sequence
    F3 = np.fft.fft(f3)
    
    # Extract the FFT results of the two sequences
    F1 = np.zeros(N, dtype=np.complex128)
    F2 = np.zeros(N, dtype=np.complex128)
    
    F1[0] = (F3[0] + np.conj(F3[0])) / 2
    F2[0] = (F3[0] - np.conj(F3[0])) / (2*1j)
    
    for m in range(1, N//2):
        F1[m] = (F3[m] + np.conj(F3[N-m])) / 2
        F2[m] = (F3[m] - np.conj(F3[N-m])) / (2*1j)
        F1[N-m] = np.conj(F1[m])
        F2[N-m] = -np.conj(F2[m])
    
    if N % 2 == 0:
        F1[N//2] = (F3[N//2] + np.conj(F3[N//2])) / 2
        F2[N//2] = (F3[N//2] - np.conj(F3[N//2])) / (2*1j)
    
    return F1, F2

# Example data for first signal pair
Fs = 1000  # Sampling frequency
T = 1 / Fs  # Sampling period
L = 1500  # Length of signal
t = np.arange(0, L) * T  # Time vector
f = Fs * np.arange(0, L) / L  # Frequency vector

# Example signals
f1_example1 = np.cos(2 * np.pi * 150 * t)
f2_example1 = np.sin(2 * np.pi * 100 * t)

# Compute FFT for first example
F1_example1, F2_example1 = fftreal(f1_example1, f2_example1)

# Plot the FFT results for first example
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(f, np.abs(F1_example1))
plt.title('Fx (Example 1)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.subplot(1, 2, 2)
plt.plot(f, np.abs(F2_example1))
plt.title('Fy (Example 1)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()

# Example data for second signal pair
f1_example2 = np.cos(2 * np.pi * 250 * t)
f2_example2 = np.sin(2 * np.pi * 200 * t)

# Compute FFT for second example
F1_example2, F2_example2 = fftreal(f1_example2, f2_example2)

# Plot the FFT results for second example
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(f, np.abs(F1_example2))
plt.title('Fx (Example 2)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.subplot(1, 2, 2)
plt.plot(f, np.abs(F2_example2))
plt.title('Fy (Example 2)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
