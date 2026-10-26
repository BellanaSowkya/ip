!apt-get update -qq
!apt-get install -y fonts-noto-core -qq

import numpy as np
import matplotlib.pyplot as plt

# 1. Define 5x5 matrices for 5 Tamil characters
# 1 represents a filled pixel, 0 represents an empty pixel
tamil_characters = {
    "அ (A)": np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]),
    "இ (I)": np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1]
    ]),
    "உ (U)": np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0]
    ]),
    "எ (E)": np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ]),
    "க (Ka)": np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1]
    ])
}

# 2. Set up the plotting grid
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

# 3. Loop through the characters and render them as images
for ax, (name, matrix) in zip(axes, tamil_characters.items()):
    # Display the matrix as a binary image (cmap='gray_r' makes 1s dark and 0s light)
    ax.imshow(matrix, cmap='gray_r', interpolation='nearest')

    # Visual adjustments to show the 5x5 gridlines clearly
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Hide the main numerical labels for clean look
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
