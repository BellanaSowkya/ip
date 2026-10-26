import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("/content/1000236966.jpg")

if image is None:
    raise FileNotFoundError("Could not find selfie.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width, channels = image.shape


def transform_image(image, matrix):
    height, width, channels = image.shape

    center_x = width / 2
    center_y = height / 2

    y_coordinates, x_coordinates = np.indices((height, width))

    x_centered = x_coordinates - center_x
    y_centered = y_coordinates - center_y

    coordinates = np.stack(
        [x_centered.ravel(), y_centered.ravel()],
        axis=0
    )

    inverse_matrix = np.linalg.inv(matrix)

    source_coordinates = inverse_matrix @ coordinates

    source_x = np.rint(
        source_coordinates[0] + center_x
    ).astype(np.int32)

    source_y = np.rint(
        source_coordinates[1] + center_y
    ).astype(np.int32)

    output = np.zeros_like(image)

    valid_pixels = (
        (source_x >= 0) &
        (source_x < width) &
        (source_y >= 0) &
        (source_y < height)
    )

    output_coordinates = np.arange(height * width)

    output_coordinates = output_coordinates[valid_pixels]

    output_y = output_coordinates // width
    output_x = output_coordinates % width

    output[output_y, output_x] = image[
        source_y[valid_pixels],
        source_x[valid_pixels]
    ]

    return output


theta = np.radians(30)

rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
], dtype=np.float32)

rotation_image = transform_image(
    image,
    rotation_matrix
)


scale_x = 1.3
scale_y = 1.3

scaling_matrix = np.array([
    [scale_x, 0],
    [0, scale_y]
], dtype=np.float32)

scaling_image = transform_image(
    image,
    scaling_matrix
)


shear_x = 0.5
shear_y = 0

shearing_matrix = np.array([
    [1, shear_x],
    [shear_y, 1]
], dtype=np.float32)

shearing_image = transform_image(
    image,
    shearing_matrix
)


cv2.imwrite(
    "rotation.jpg",
    cv2.cvtColor(rotation_image, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    "scaling.jpg",
    cv2.cvtColor(scaling_image, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    "shearing.jpg",
    cv2.cvtColor(shearing_image, cv2.COLOR_RGB2BGR)
)


plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(rotation_image)
plt.title("Rotation")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(scaling_image)
plt.title("Scaling")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(shearing_image)
plt.title("Shearing")
plt.axis("off")

plt.tight_layout()
plt.show()
