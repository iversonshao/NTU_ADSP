import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio
from io import BytesIO
from PIL import Image
import os
import datetime

# Directory containing the BMP images
image_dir = 'Final/dataset'

# Prepare a list to store the results
results = []

# Size of the uncompressed data
uncompressed_size = 28800  # bytes

# Compression and saving images
for i in range(1, 11):
    # Paths for the original and compressed images
    bmp_image_path = os.path.join(image_dir, f'{i:02d}.bmp')
    jpg_image_path = f"./{i:02d}.jpg"
    
    # Open the original BMP image
    im1 = Image.open(bmp_image_path)

    # Creating an empty string buffer for JPEG compression
    buffer = BytesIO()
    im1.save(buffer, "JPEG", quality=60)

    # Write the buffer to a file
    with open(jpg_image_path, "wb") as handle:
        handle.write(buffer.getbuffer())

    # Open the compressed image for comparison
    im2 = Image.open(jpg_image_path)
    img2 = img_as_float(im2)

    # Convert original image to float for comparison
    img1 = img_as_float(im1)

    # Calculate SSIM
    ssim_value = ssim(img1, img2, data_range=img2.max() - img2.min(), multichannel=True)

    # Calculate PSNR
    psnr_value = peak_signal_noise_ratio(img1, img2)
    
    # Calculate the size of the compressed images
    compressed_size = os.path.getsize(jpg_image_path)
    n_compression = compressed_size / uncompressed_size

    # Append the results to the list
    results.append((f'{i:02d}.png', ssim_value, psnr_value, n_compression))

# Convert the results to a DataFrame for better visualization
df_results = pd.DataFrame(results, columns=['Image', 'SSIM', 'PSNR', 'N Compression'])

# Calculate the average SSIM, PSNR, and N Compression
average_row = pd.DataFrame([['Average', df_results['SSIM'].mean(), df_results['PSNR'].mean(), df_results['N Compression'].mean()]], columns=df_results.columns)
df_results = pd.concat([df_results, average_row], ignore_index=True)

# Print the DataFrame
print(df_results)

# Generate a unique filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'compression_results_{timestamp}.csv'

# Save the results to a CSV file
df_results.to_csv(csv_filename, index=False)

# Display the DataFrame as a table using matplotlib with a larger font size
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size as needed
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=df_results.values, colLabels=df_results.columns, cellLoc='center', loc='center')

# Adjust the font size
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)

# Adjust the column widths
for key, cell in the_table.get_celld().items():
    cell.set_height(0.1)
    if key[0] == 0:
        cell.set_fontsize(12)  # Header font size
    else:
        cell.set_fontsize(12)  # Body font size


plt.show()
