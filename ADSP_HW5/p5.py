import numpy as np

# Constants
N = 12

# Compute the unit root
omega = np.exp(2j * np.pi / N)

# Initialize the FFT matrix
fft_matrix = np.zeros((N, N), dtype=complex)

# Fill the FFT matrix with powers of the unit root omega
for j in range(N):
    for k in range(N):
        fft_matrix[j, k] = omega ** (j * k)

# Function to format complex numbers
def format_complex(c, precision=2):
    return f"{c.real:.{precision}e}+{c.imag:.{precision}e}j"

# Print the FFT matrix with formatted output
print("FFT Matrix:")
for row in fft_matrix:
    formatted_row = [format_complex(c) for c in row]
    print("[" + ", ".join(formatted_row) + "]")
