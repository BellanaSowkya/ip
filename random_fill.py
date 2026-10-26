import numpy as np
import matplotlib.pyplot as plt

rows = 3
cols = 3

while True:
    matrix = np.random.randint(0, 2, size=(rows, cols))

    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    plt.title("Randomly Generated 3 × 3 Black-and-White Image", fontsize=14, fontweight="bold")
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    for i in range(rows):
        for j in range(cols):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="red", fontsize=14)

    plt.xticks(range(cols))
    plt.yticks(range(rows))
    plt.grid(True, color="blue", linewidth=1)

    plt.show(block=False)
    plt.pause(1)

    response = input("Is this a sensible image? (y/n): ").lower()

    plt.close()

    if response == "y":
        print("Sensible image found!")
        print(matrix)
        break
