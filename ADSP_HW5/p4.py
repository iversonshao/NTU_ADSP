import numpy as np

v1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
v2 = np.array([1, 1, 1, 1, -1, -1, -1, -1,  1, 1, 1, 1, -1, -1, -1, -1])
v3 = np.array([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])

vk = [v1, -v1, v1]

# Custom print function to remove commas
def custom_print(data):
    if isinstance(data, (int, np.integer)):
        print(data)
    else:
        print(' '.join(map(str, data)))

# Function to compute inner product
def inner_product(v1, v2):
    return np.dot(v1, v2)

# Example usage
# custom_print(vk)
# custom_print([v2, v2, -v2])
# custom_print([v1+v2-v3, -v1+v2+v3, v1-v2+v3])
result = inner_product(v1-v2+v3, v3)
custom_print(result)
