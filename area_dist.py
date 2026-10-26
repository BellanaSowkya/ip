import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab.patches import cv2_imshow

images = [
    ("4-1.jpg", 10),
    ("4-2.jpg", 15),
    ("4-3.jpg", 20),
    ("4-4.jpg", 25),
    ("4-5.jpg", 30),
    ("4-6.jpg", 35),
    ("4-7.jpg", 40),
    ("4-8.jpg", 45),
    ("4-9.jpg", 50),
    ("4-10.jpg", 55)
]

os.makedirs("output", exist_ok=True)

distances = []
percentage_areas = []

fixed_width = 600
fixed_height = 450
heading_height = 70

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
thickness = 2

for filename, distance in images:

    image = cv2.imread(filename)

    if image is None:
        print("Could not read:", filename)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=500
    )

    if circles is None:
        print("Coin not detected:", filename)
        continue

    circles = np.round(circles[0, :]).astype(int)

    x, y, r = max(circles, key=lambda c: c[2])

    coin_area = np.pi * r * r

    height, width = gray.shape
    image_area = width * height

    percentage = (coin_area / image_area) * 100

    distances.append(distance)
    percentage_areas.append(percentage)

    print(
        f"{filename}: "
        f"Distance = {distance} cm, "
        f"Radius = {r} pixels, "
        f"Coin Area = {coin_area:.2f} pixels², "
        f"Percentage = {percentage:.4f}%"
    )

    output = image.copy()

    cv2.circle(
        output,
        (x, y),
        r,
        (0, 255, 0),
        3
    )

    cv2.circle(
        output,
        (x, y),
        2,
        (0, 0, 255),
        3
    )

    # ---------------------------------------
    # Resize image BEFORE adding any text
    # ---------------------------------------

    output = cv2.resize(
        output,
        (fixed_width, fixed_height)
    )

    # ---------------------------------------
    # Create fixed-size heading
    # ---------------------------------------

    heading = np.zeros(
        (heading_height, fixed_width, 3),
        dtype=np.uint8
    )

    text_size = cv2.getTextSize(
        filename,
        font,
        font_scale,
        thickness
    )[0]

    text_x = (fixed_width - text_size[0]) // 2
    text_y = (heading_height + text_size[1]) // 2

    cv2.putText(
        heading,
        filename,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # ---------------------------------------
    # Add distance and area AFTER resizing
    # ---------------------------------------

    cv2.putText(
        output,
        f"Distance: {distance} cm",
        (20, 40),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"Area: {percentage:.2f}%",
        (20, 80),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )

    output_with_heading = np.vstack(
        (heading, output)
    )

    # ---------------------------------------
    # Save processed image
    # ---------------------------------------

    output_filename = os.path.join(
        "output",
        f"processed_{filename}"
    )

    cv2.imwrite(output_filename, output_with_heading)

    print("Saved:", output_filename)

    cv2_imshow(output_with_heading)

# ---------------------------------------
# Plot graph
# ---------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    distances,
    percentage_areas,
    label="Measured values"
)

plt.plot(
    distances,
    percentage_areas
)

plt.xlabel("Capturing Distance (cm)")
plt.ylabel("Percentage Area Occupied (%)")
plt.title("Capturing Distance vs Percentage Area Occupied by Coin")

plt.grid(True)

plt.legend()

plt.savefig(
    "output/distance_vs_area.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
