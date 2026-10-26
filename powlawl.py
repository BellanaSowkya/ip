import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "/content/1000236966.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gamma_values = [0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0]

gamma_images = []

for gamma in gamma_values:
    normalized_image = image / 255.0

    gamma_image = 255 * (normalized_image ** gamma)

    gamma_image = np.clip(gamma_image, 0, 255).astype(np.uint8)

    gamma_images.append(gamma_image)

    output_filename = f"selfie_gamma_{gamma}.jpg"
    cv2.imwrite(
        output_filename,
        cv2.cvtColor(gamma_image, cv2.COLOR_RGB2BGR)
    )

plt.figure(figsize=(16, 10))

plt.subplot(2, 4, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

for index, gamma in enumerate(gamma_values):
    plt.subplot(2, 4, index + 2)
    plt.imshow(gamma_images[index])
    plt.title(f"Gamma = {gamma}")
    plt.axis("off")

plt.tight_layout()
plt.show()
