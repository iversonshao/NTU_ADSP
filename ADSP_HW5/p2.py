import math
import matplotlib.pyplot as plt

# Define the formula function
def f(L, N, M):
    term1 = N / L
    term2 = 3 * (L + M - 1)
    term3 = math.log(L + M - 1, 2) + 1
    return term1 * term2 * term3

# Initialize variables
N = 1200
M = 2
L_min = 1
L_max = 600

# Record all L values and their corresponding function values
L_values = list(range(L_min, L_max + 1))
f_values = [f(L, N, M) for L in L_values]

# Find the minimum value and corresponding L
min_value = min(f_values)
min_L = L_values[f_values.index(min_value)]

# Print the result
print(f"Minimum value of f(L) is {min_value:.2f} at L = {min_L}")

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(L_values, f_values, label='f(L)')
plt.scatter(min_L, min_value, color='red', label=f'Minimum at L={min_L}, f(L)={min_value:.2f}')
plt.xlabel('L')
plt.ylabel('f(L)')
plt.title('Function f(L) vs L')
plt.legend()
plt.grid(True)
plt.show()
