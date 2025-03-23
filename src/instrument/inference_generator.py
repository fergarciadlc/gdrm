import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.generator import Generator

NOISE_DIM = 5


# Function to load generator checkpoint
def load_generator(checkpoint_path, generator: Generator, device="cpu"):
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    return generator


def realify_genered_pattern(fake_pattern: torch.Tensor):
    return torch.where(
        fake_pattern <= 0, torch.tensor(-1.0, device=fake_pattern.device), fake_pattern
    )


def load_genre_classes(genre_classes_path):
    """loads classes from json file"""
    with open(genre_classes_path, "r") as f:
        genre_classes = json.load(f)
    return genre_classes


def run_inference(
    checkpoint_path: str,
    generator: nn.Module,
    genre_str: str = "funk",
    genre_mapping_path: str = "bar_arrays/genre_mapping.json",
    bpm: float = 120.0,
    device: str = "cpu",
):
    genre_mapping = load_genre_classes(genre_mapping_path)
    genre_idx = genre_mapping[genre_str]
    normalised_bpm = 2 * ((bpm - 60) / (200 - 60)) - 1

    # Load the trained generator
    generator = load_generator(
        checkpoint_path=checkpoint_path, generator=generator, device=device
    ).to(device)

    # Define your input dimensions
    noise_dim = NOISE_DIM

    # Example inputs
    batch_size = 1
    noise = torch.randn(batch_size, noise_dim).to(device)
    genre = (
        F.one_hot(
            torch.tensor(genre_idx, dtype=torch.long),
            num_classes=len(genre_mapping),
        )
        .unsqueeze(0)
        .to(device)
    )
    bpm = torch.tensor([[normalised_bpm]]).to(device)  # Example normalized bpm

    # Generate sample
    with torch.no_grad():
        generated_output = generator(noise, genre, bpm)

    generated_output = generated_output[0].cpu().numpy()

    return generated_output

