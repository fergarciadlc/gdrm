"""
Quick plotting utility for visualizing a bar array saved as a .npy file.
The array is assumed to have dimensions 10 (drum families) x 48 (grid locations).

Usage:
    Run the script to load "bar_arrays/bar_1.npy" and display an image plot.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_bar_array(filepath: str) -> None:
    """
    Loads the numpy array from the given filepath and plots it.

    Parameters:
        filepath (str): Path to the .npy file to be plotted.
    """
    # Load the numpy array.
    bar_array = np.load(filepath)

    # Create a plot of the bar array.
    plt.figure(figsize=(10, 4))
    plt.imshow(bar_array, cmap="viridis", aspect="auto")
    plt.colorbar(label="Velocity Sum")
    plt.title(f"Bar Array: {os.path.basename(filepath)}")
    plt.xlabel("Normalized Onset Index")
    plt.ylabel("Drum Family (index)")

    # Configure ticks.
    plt.xticks(range(bar_array.shape[1]))
    plt.yticks(range(bar_array.shape[0]), range(1, bar_array.shape[0] + 1))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define the expected path to the bar array.
    filepath = os.path.join("bar_arrays", "bar_3.npy")

    if not os.path.exists(filepath):
        print(
            "Bar file not found. Make sure to run the preprocessing script first to generate bar arrays."
        )
    else:
        plot_bar_array(filepath)
