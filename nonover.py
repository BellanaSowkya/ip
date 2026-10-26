# CONVOLUTION ON NON-OVERLAPPING CIRCLES
# Roll number divisible by 2
# Run this entire cell in Google Colab

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# --------------------------------------------------
# 1. GENERATE ORIGINAL IMAGE WITH NON-OVERLAPPING CIRCLES
# --------------------------------------------------

image_size = 256
circle_radius = 9
spacing = 32

# Black background
image = np.zeros((image_size, image_size), dtype=np.float32)

# Place equally sized, non-overlapping white circles
for y in range(16, image_size - 10, spacing):
    for x in range(16, image_size - 10, spacing):

        Y, X = np.ogrid[:image_size, :image_size]

        mask = (Y - y)**2 + (X - x)**2 <= circle_radius**2
        image[mask] = 255

# --------------------------------------------------
# 2. DEFINE CONVOLUTION KERNEL
# --------------------------------------------------

# Laplacian kernel for detecting circle boundaries
kernel = np.array([
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
], dtype=np.float32)

# --------------------------------------------------
# 3. APPLY CONVOLUTION
# --------------------------------------------------

convolution_output = convolve2d(
    image,
    kernel,
    mode='same',
    boundary='fill',
    fillvalue=0
)

# Absolute values for clearer edge visualization
output_display = np.abs(convolution_output)

# --------------------------------------------------
# 4. DISPLAY ORIGINAL IMAGE AND CONVOLUTION OUTPUT
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original circles image
axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Original Image - Non-Overlapping Circles",
                  fontsize=13, fontweight='bold')
axes[0].axis('off')

# Convolution output
im = axes[1].imshow(output_display, cmap='hot')
axes[1].set_title("Convolution Output - Edge Detection",
                  fontsize=13, fontweight='bold')
axes[1].axis('off')

plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5. DISPLAY RESULTS
# --------------------------------------------------

print("Original Image Shape:", image.shape)
print("Convolution Output Shape:", convolution_output.shape)

print("\nConvolution Kernel Used:")
print(kernel)

print("\nSample Convolution Output (10x10 region):")
print(convolution_output[10:20, 10:20])

print("\nConvolution completed successfully!")




























