import cv2
from matplotlib import pyplot as plt
from math import sqrt
import numpy as np

# Load the original images
img_original1 = cv2.imread("ADSP_HW4/baboon.jpg", cv2.IMREAD_GRAYSCALE)
img_comparison1 = cv2.imread("ADSP_HW4/bell_peper.jpg", cv2.IMREAD_GRAYSCALE)

# Resize the images to a common size
img_original1 = cv2.resize(img_original1, (256, 256))
img_comparison1 = cv2.resize(img_comparison1, (256, 256))

# Display one of the original images
img_comparison2 = img_original1 * 0.5 + 255.5 * 0.5

img_y = img_comparison1
img_z = img_comparison2

img_y = np.uint8(img_y)
img_z = np.uint8(img_z)

# Calculate SSIM between the original images and the comparison image
M, N = img_original1.shape
L = 255
c1 = 1 / sqrt(L)
c2 = 1 / sqrt(L)

u_x = img_original1.mean(axis=0).mean(axis=0)
u_y = img_y.mean(axis=0).mean(axis=0)
u_z = img_z.mean(axis=0).mean(axis=0)

sigmax = np.sum(np.power(img_original1 - u_x, 2)) * (1 / (M * N))
sigmay = np.sum(np.power(img_y - u_y, 2)) * (1 / (M * N))
sigmaz = np.sum(np.power(img_z - u_z, 2)) * (1 / (M * N))

sigma_xy = np.sum((img_original1 - u_x) * (img_y - u_y)) * (1 / (M * N))
sigma_xz = np.sum((img_original1 - u_x) * (img_z - u_z)) * (1 / (M * N))

SSIM_1_2 = (2 * u_x * u_y + (c1 * L) ** 2) * (2 * sigma_xy + (c2 * L) ** 2) * (
        (u_x ** 2 + u_y ** 2 + (c1 * L) ** 2) ** (-1)) * ((sigmax + sigmay + (c2 * L) ** 2) ** (-1))
SSIM_1_3 = (2 * u_x * u_z + (c1 * L) ** 2) * (2 * sigma_xz + (c2 * L) ** 2) * (
        (u_x ** 2 + u_z ** 2 + (c1 * L) ** 2) ** (-1)) * ((sigmax + sigmaz + (c2 * L) ** 2) ** (-1))


# Display images and SSIM values
plt.figure(figsize=(12, 5))
plt.subplot(1, 4, 1)
plt.imshow(img_original1, cmap='gray')
plt.title('Original Image 1')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(img_comparison1, cmap='gray')
plt.title(f'Comparison Image1\nSSIM = {SSIM_1_2:.4f}')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(img_original1, cmap='gray')
plt.title('Original Image 1')
plt.axis('off')

img_z1 = cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB)
plt.subplot(1, 4, 4)
plt.imshow(img_z1, cmap='gray')  
plt.title(f'Comparison Image2\nSSIM = {SSIM_1_3:.4f}')  
plt.axis('off')

plt.tight_layout()
plt.show()
