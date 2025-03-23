import os

import numpy as np
import torch
import torch.nn as nn

from src.model.generator import Generator


# Function to load generator checkpoint
def load_generator(checkpoint_path, generator: Generator, device="cpu"):
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    return generator


def realify_genered_pattern(fake_pattern: torch.Tensor):
    return torch.where(
        fake_pattern <= 0, torch.tensor(-1.0, device=fake_pattern.device), fake_pattern
    )


# Example inference
if __name__ == "__main__":
    device = "mps"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(
        input_dim=24,
        output_dim=1920,
        img_shape=(10, 192),
    )
    checkpoint_path = (
        "checkpoints/generator_epoch_100.pth"  # Update with your checkpoint path
    )
    genre_idx = 3
    bpm = 80.0
    normalised_bpm = 2 * ((bpm - 60) / (200 - 60)) - 1

    # Load the trained generator
    generator = load_generator(
        checkpoint_path=checkpoint_path, generator=generator, device=device
    ).to(device)

    # Define your input dimensions
    noise_dim = 5
    genre_dim = 18

    # Example inputs
    batch_size = 1
    noise = torch.randn(batch_size, noise_dim).to(device)
    genre = torch.zeros(batch_size, genre_dim).to(device)
    genre[:, genre_idx] = 1  # Example genre, first genre selected (one-hot)
    bpm = torch.tensor([[normalised_bpm]]).to(device)  # Example normalized bpm

    # Generate sample
    with torch.no_grad():
        generated_output = realify_genered_pattern(generator(noise, genre, bpm))

    generated_output = generated_output[0].cpu().numpy()

    print("Generated output shape:", generated_output.shape)
    print("Generated output:", generated_output)
    # save the generated output as a .npy file
    np.save("generated_output.npy", generated_output)
