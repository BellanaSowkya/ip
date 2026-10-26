import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "/content/bird.jpeg"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

rows, cols = image.shape

random_row = np.random.randint(2, rows - 2)
random_col = np.random.randint(2, cols - 2)

pixel_value = image[random_row, random_col]

eight_neighbourhood = image[
    random_row - 1:random_row + 2,
    random_col - 1:random_col + 2
]

sixteen_neighbourhood = image[
    random_row - 2:random_row + 3,
    random_col - 2:random_col + 3
]

print("Image Matrix Shape:", image.shape)

print("\nRandom Pixel Position:")
print("Row:", random_row)
print("Column:", random_col)

print("\nRandom Pixel Value:")
print(pixel_value)

print("\n8-Neighbourhood Submatrix:")
print(eight_neighbourhood)

print("\n16-Neighbourhood Submatrix:")
print(sixteen_neighbourhood)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(eight_neighbourhood, cmap="gray")
plt.title("8-Neighbourhood")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sixteen_neighbourhood, cmap="gray")
plt.title("16-Neighbourhood")
plt.axis("off")

plt.tight_layout()
plt.show()
