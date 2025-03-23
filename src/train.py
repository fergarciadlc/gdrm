"""
Training loop for a conditional WGAN-GP with auxiliary classifiers for genre and BPM.
Each sample is a 2D array of shape (10,196) with values in [-1, 1].
Conditions:
  - Genre: provided as a one-hot vector (dim 18)
  - BPM: a continuous value (normalized) with shape (1,)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import autograd
from tqdm import tqdm

from model.discriminator import Discriminator
from model.generator import Generator
from dataloader import GrooveDataset

# Hyperparameters
BATCH_SIZE = 32
NOISE_DIM = 5
GENRE_DIM = 18     # one-hot genre vector dimension
BPM_DIM = 1        # one additional value for BPM
NUM_EPOCHS = 100
LEARNING_RATE = 0.0002

# The generated “drum pattern” is 10x192 = 1920 numbers.
IMG_ROWS = 10
IMG_COLS = 192
IMG_DIM = IMG_ROWS * IMG_COLS

# Loss functions for auxiliary tasks:
classification_loss_fn = nn.CrossEntropyLoss()
bpm_loss_fn = nn.MSELoss()

# Gradient penalty coefficient
LAMBDA_GP = 10

# Helper function for gradient penalty
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, device=device)
    # Match dimensions for interpolation (keep dim for each sample)
    epsilon = epsilon.expand_as(real_samples)
    # Create interpolated samples
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)

    # Forward pass
    interpolated_validity, _, _ = discriminator(interpolated)
    # For each sample, take gradients of outputs with respect to interpolated inputs
    grad_outputs = torch.ones(interpolated_validity.size(), device=device)
    gradients = autograd.grad(
        outputs=interpolated_validity,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Compute the Euclidean norm for each gradient in the batch
    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    # Compute penalty as (||gradients||2 - 1)^2
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty

def realify_genered_pattern(fake_pattern: torch.Tensor):
    return torch.where(fake_pattern <= 0, torch.tensor(-1.0), fake_pattern)


def train(generator: Generator, discriminator: Discriminator, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(tqdm(dataloader)):
            real_data = batch["data"].to(device)                     # shape: (batch, 10,192)
            genre_data = batch["metadata"]["genre"].to(device)         # shape: (batch, 18) one-hot
            bpm_data = batch["metadata"]["bpm"].unsqueeze(dim=1).to(device)  # shape: (batch, 1); normalized

            current_batch = real_data.size(0)

            # Convert one-hot genre to label indices for CrossEntropyLoss.
            real_genre_labels = genre_data.argmax(dim=1)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Flatten real data: shape (batch, 10,192) → (batch,1920)
            real_data_flat = real_data.view(current_batch, -1)
            # Forward pass for real samples:
            real_validity, real_genre_pred, real_bpm_pred = discriminator(real_data_flat)
            # Wasserstein loss for real samples (we want high scores): -E[D(real)]
            d_real_loss = - real_validity.mean()
            # Auxiliary losses for real samples:
            real_class_loss = classification_loss_fn(real_genre_pred, real_genre_labels)
            real_bpm_loss = bpm_loss_fn(real_bpm_pred, bpm_data)

            # Generate fake samples:
            noise = torch.randn(current_batch, NOISE_DIM, device=device)
            fake_data = realify_genered_pattern(generator(noise, genre_data, bpm_data))  # shape: (batch,10,192)
            fake_data_flat = fake_data.view(current_batch, -1)
            # Forward pass for fake samples:
            fake_validity, fake_genre_pred, fake_bpm_pred = discriminator(fake_data_flat.detach())
            # Wasserstein loss for fake samples (we want low scores): E[D(fake)]
            d_fake_loss = fake_validity.mean()
            # Auxiliary losses for fake samples (forcing fake samples to match conditioning):
            fake_class_loss = classification_loss_fn(fake_genre_pred, real_genre_labels)
            fake_bpm_loss = bpm_loss_fn(fake_bpm_pred, bpm_data)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_data_flat.data, fake_data_flat.data, device)

            # Total discriminator loss:
            # for WGAN: D loss = E[D(fake)] - E[D(real)] + lambda_gp * gradient_penalty
            # plus auxiliary losses from both real and fake samples.
            d_adv_loss = d_fake_loss + d_real_loss
            aux_loss = 0.5 * (real_class_loss + fake_class_loss) + 0.5 * (real_bpm_loss + fake_bpm_loss)
            d_loss = d_adv_loss + LAMBDA_GP * gradient_penalty + aux_loss

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Generate fake samples with the same conditioning:
            gen_data = realify_genered_pattern(generator(noise, genre_data, bpm_data))
            gen_data_flat = gen_data.view(current_batch, -1)
            validity, genre_pred, bpm_pred = discriminator(gen_data_flat)

            # Generator adversarial loss: we want D(gen_data) to be as high as possible,
            # so for WGAN, g_adv_loss = -E[D(gen_data)]
            g_adv_loss = - validity.mean()
            g_class_loss = classification_loss_fn(genre_pred, real_genre_labels)
            g_bpm_loss = bpm_loss_fn(bpm_pred, bpm_data)
            g_loss = g_adv_loss + g_class_loss + g_bpm_loss

            g_loss.backward()
            optimizer_G.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

if __name__ == "__main__":

    # Create generator and discriminator instances.
    # Note: The generator's input_dim is now noise_dim + genre_dim + bpm_dim.
    generator = Generator(input_dim=NOISE_DIM + GENRE_DIM + BPM_DIM, output_dim=IMG_DIM, img_shape=(IMG_ROWS, IMG_COLS))
    discriminator = Discriminator(input_dim=IMG_DIM, num_genres=GENRE_DIM)

    dataset_directory = "bar_arrays"

    print("Loading dataset")
    # Make sure your GrooveDataset returns a dictionary with "data" and "metadata" that now also has "bpm".
    dataset = GrooveDataset(root_dir=dataset_directory)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("Begin training")
    train(generator, discriminator, dataloader)
