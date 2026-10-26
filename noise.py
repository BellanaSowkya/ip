import cv2
import numpy as np
import matplotlib.pyplot as plt

original = cv2.imread("/content/1000236966.jpg")

if original is None:
    raise FileNotFoundError("selfie.jpg not found.")

original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

original_float = original.astype(np.float32)

mean = 110
sigma = 5

noise = np.random.normal(
    mean,
    sigma,
    original_float.shape
).astype(np.float32)

noisy = original_float + noise

noisy = np.clip(noisy, 0, 255)

noisy_uint8 = noisy.astype(np.uint8)

kernel = np.ones((5, 5), dtype=np.float32) / 25

print("5 × 5 Averaging Kernel:")
print(kernel)


recovered = cv2.filter2D(
    noisy_uint8,
    -1,
    kernel
)

recovered_float = recovered.astype(np.float32)

mse = np.mean(
    (original_float - recovered_float) ** 2
)

print("\nMean Squared Error (MSE):", mse)

noisy_rgb = cv2.cvtColor(
    noisy_uint8,
    cv2.COLOR_BGR2RGB
)

recovered_rgb = cv2.cvtColor(
    recovered,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_rgb)
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(recovered_rgb)
plt.title("Recovered Image")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite(
    "noisy_image.jpg",
    noisy_uint8
)

cv2.imwrite(
    "recovered_image.jpg",
    recovered
)

print("\nImages saved:")
print("noisy_image.jpg")
print("recovered_image.jpg")
