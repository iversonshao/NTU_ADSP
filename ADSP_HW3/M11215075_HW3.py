import numpy as np
import matplotlib.pyplot as plt

def rgb2ycbcr(rgb):
    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.169, -0.331, 0.5],
                       [0.5, -0.419, -0.081]])
    
    rgb = rgb.astype(np.float32) / 255.0
    
    ycbcr = np.dot(rgb, matrix.T)
    ycbcr[:, :, 0] += 16 / 255  
    ycbcr[:, :, 1:] += 128 / 255  
    
    return ycbcr

def ycbcr2rgb(ycbcr):
    matrix = np.array([[1, 0, 1.402],
                       [1, -0.344, -0.714],
                       [1, 1.772, 0]])
    
    ycbcr = ycbcr.copy()
    ycbcr[:, :, 0] -= 16 / 255
    ycbcr[:, :, 1:] -= 128 / 255
    
    rgb = np.dot(ycbcr, matrix.T)
    rgb = np.clip(rgb, 0, 1)  
    
    return rgb

def compress_420(img):
    ycbcr = rgb2ycbcr(img)
    
    h, w, c = ycbcr.shape
    cb = ycbcr[::2, ::2, 1]
    cr = ycbcr[::2, ::2, 2]
    
    cb_up = np.repeat(np.repeat(cb, 2, axis=1), 2, axis=0)
    cr_up = np.repeat(np.repeat(cr, 2, axis=1), 2, axis=0)
    
    ycbcr_compressed = np.zeros_like(ycbcr)
    ycbcr_compressed[:, :, 0] = ycbcr[:, :, 0]
    ycbcr_compressed[:, :, 1] = cb_up
    ycbcr_compressed[:, :, 2] = cr_up
    
    rgb_compressed = ycbcr2rgb(ycbcr_compressed)
    rgb_compressed = (rgb_compressed * 255).astype(np.uint8)
    
    return rgb_compressed
#can change the path to the image you want to compress
img = plt.imread('ADSP_HW3\\test.jpg')
compressed_img = compress_420(img)

# Save the compressed image
plt.imsave('compressed_image.jpg', compressed_img)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(compressed_img)
plt.title('Compressed Image')
plt.axis('off')
plt.show()
