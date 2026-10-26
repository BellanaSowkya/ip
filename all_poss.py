import numpy as np
import matplotlib.pyplot as plt

rows = 3
cols = 2
num_pixels = rows * cols
num_images = 2 ** num_pixels

fig, axes = plt.subplots(8, 8, figsize=(12, 14))

fig.suptitle(
    "All Possible Images of a 3 × 2 Black-and-White Camera",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

for i, ax in enumerate(axes.flat):
    binary = format(i, f"0{num_pixels}b")
    image = np.array([int(bit) for bit in binary]).reshape(rows, cols)

    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"Image {i + 1}\n{binary}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
