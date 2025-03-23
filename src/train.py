"""
Training loop for a conditional WGAN-GP with auxiliary classifiers for genre and BPM.
Each sample is a 2D array of shape (10,196) with values in [-1, 1].

Conditions:
  - Genre: provided as a one-hot vector (dim 18)
  - BPM: a continuous value (normalized) with shape (1,)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import autograd
from tqdm import tqdm
import subprocess
import sys

import logging
from torch.utils.tensorboard.writer import SummaryWriter

from model.discriminator import Discriminator
from model.generator import Generator
from dataloader import GrooveDataset

torch.multiprocessing.set_sharing_strategy("file_system")

# ------------------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------------------
BATCH_SIZE = 32
NOISE_DIM = 5
GENRE_DIM = 18     # one-hot genre vector dimension
BPM_DIM = 1        # one additional value for BPM
NUM_EPOCHS = 100
LEARNING_RATE = 0.0002

# Generated drum pattern dimensions: 10x192 = 1920 numbers.
IMG_ROWS = 10
IMG_COLS = 192
IMG_DIM = IMG_ROWS * IMG_COLS

# Gradient penalty coefficient.
LAMBDA_GP = 50

# ------------------------------------------------------------------------------
# Auxiliary Loss Weighting Hyperparameters:
# ------------------------------------------------------------------------------

#   For Discriminator:
LAMBDA_CLASS_D = 5.0   # Weight for the genre classification loss.
LAMBDA_BPM_D   = 1.0   # Weight for the BPM regression loss.

#   For Generator:
LAMBDA_CLASS_G = 5.0   # Weight for the generator's genre classification loss.
LAMBDA_BPM_G   = 3.0   # Weight for the generator's BPM regression loss.

# Introduce number of critic updates per generator update.
N_CRITIC = 3

# Loss functions for auxiliary tasks:
classification_loss_fn = nn.CrossEntropyLoss()
bpm_loss_fn = nn.MSELoss()

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, device=device)
    epsilon = epsilon.expand_as(real_samples)
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)

    interpolated_validity, _, _ = discriminator(interpolated)
    grad_outputs = torch.ones(interpolated_validity.size(), device=device)
    gradients = autograd.grad(
        outputs=interpolated_validity,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty


# ------------------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------------------
def train(generator: Generator, discriminator: Discriminator, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Initialize TensorBoard SummaryWriter.
    writer = SummaryWriter(log_dir='./runs/gdrm_experiment')

    # Ensure a directory exists for saving checkpoints.
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(tqdm(dataloader)):
            real_data = batch["data"].to(device)                     # shape: (batch, 10, 192)
            genre_data = batch["metadata"]["genre"].to(device)         # shape: (batch, 18) one-hot encoding
            bpm_data = batch["metadata"]["bpm"].unsqueeze(dim=1).to(device)  # shape: (batch, 1): normalized BPM

            current_batch = real_data.size(0)
            # Convert one-hot genre into label indices for CrossEntropyLoss.
            real_genre_labels = genre_data.argmax(dim=1)

            # Initialize accumulators for additional loss components for the discriminator.
            d_loss_total = 0.0
            d_adv_loss_sum = 0.0
            d_grad_penalty_sum = 0.0
            d_class_loss_sum = 0.0   # Sum for both real and fake classification losses.
            d_bpm_loss_sum = 0.0     # Sum for both real and fake BPM regression losses.

            # ------------------------------------------------------------------
            # Multiple Discriminator (Critic) Updates
            # ------------------------------------------------------------------
            for _ in range(N_CRITIC):
                optimizer_D.zero_grad()

                # Flatten real data: (batch, 10,192) → (batch,1920)
                real_data_flat = real_data.view(current_batch, -1)
                # Forward pass on real samples.
                real_validity, real_genre_pred, real_bpm_pred = discriminator(real_data_flat)
                # Adversarial loss for real samples: aim for high values.
                d_real_loss = - real_validity.mean()
                # Auxiliary losses for real samples.
                real_class_loss = classification_loss_fn(real_genre_pred, real_genre_labels)
                real_bpm_loss = bpm_loss_fn(real_bpm_pred, bpm_data)

                # Generate fake samples with fresh noise for each critic update.
                noise = torch.randn(current_batch, NOISE_DIM, device=device)
                # Concatenate noise with the conditioning data.
                fake_data = generator(noise, genre_data, bpm_data)   # shape: (batch,10,192)
                fake_data_flat = fake_data.view(current_batch, -1)
                # Forward pass on fake samples.
                fake_validity, fake_genre_pred, fake_bpm_pred = discriminator(fake_data_flat.detach())
                # Adversarial loss for fake samples: aim for low values.
                d_fake_loss = fake_validity.mean()
                # Auxiliary losses for fake samples.
                fake_class_loss = classification_loss_fn(fake_genre_pred, real_genre_labels)
                fake_bpm_loss = bpm_loss_fn(fake_bpm_pred, bpm_data)

                # Calculate gradient penalty.
                gradient_penalty = compute_gradient_penalty(discriminator, real_data_flat.data, fake_data_flat.data, device)

                # Total adversarial loss for discriminator.
                d_adv_loss = d_fake_loss + d_real_loss

                # Compute weighted auxiliary losses.
                aux_loss = (LAMBDA_CLASS_D * (real_class_loss + fake_class_loss) +
                            LAMBDA_BPM_D   * (real_bpm_loss  + fake_bpm_loss))

                # Total discriminator loss.
                d_loss = d_adv_loss + LAMBDA_GP * gradient_penalty + aux_loss

                d_loss.backward()
                optimizer_D.step()

                # Accumulate losses over the N_CRITIC iterations.
                d_loss_total      += d_loss.item()
                d_adv_loss_sum    += d_adv_loss.item()
                d_grad_penalty_sum += gradient_penalty.item()
                d_class_loss_sum  += (real_class_loss + fake_class_loss).item()
                d_bpm_loss_sum    += (real_bpm_loss + fake_bpm_loss).item()

            # ------------------------------------------------------------------
            # Generator Update (use fresh noise and conditioning)
            # ------------------------------------------------------------------
            optimizer_G.zero_grad()
            # Sample a new noise vector; note this is distinct from the one used in the last D update.
            noise_gen = torch.randn(current_batch, NOISE_DIM, device=device)
            gen_data = generator(noise_gen, genre_data, bpm_data)
            gen_data_flat = gen_data.view(current_batch, -1)
            validity, genre_pred, bpm_pred = discriminator(gen_data_flat)

            # Generator adversarial loss: aim to maximize discriminator scores.
            g_adv_loss = - validity.mean()
            # Auxiliary losses for the generator: push generated samples to match conditions.
            g_class_loss = classification_loss_fn(genre_pred, real_genre_labels)
            g_bpm_loss = bpm_loss_fn(bpm_pred, bpm_data)

            # Total generator loss including explicit weighting.
            g_loss = g_adv_loss + (LAMBDA_CLASS_G * g_class_loss) + (LAMBDA_BPM_G * g_bpm_loss)

            g_loss.backward()
            optimizer_G.step()

            if i % 50 == 0:
                iteration = epoch * len(dataloader) + i
                avg_d_loss         = d_loss_total      / N_CRITIC
                avg_d_adv_loss     = d_adv_loss_sum    / N_CRITIC
                avg_d_grad_penalty = d_grad_penalty_sum / N_CRITIC
                avg_d_class_loss   = d_class_loss_sum  / N_CRITIC
                avg_d_bpm_loss     = d_bpm_loss_sum    / N_CRITIC

                # Update tqdm bar with the average losses
                tqdm.write(f"[Epoch {epoch + 1}/{NUM_EPOCHS}] [Batch {i}/{len(dataloader)}] "
                           f"[D loss: {avg_d_loss:.4f}] [G loss: {g_loss.item():.4f}]")

                # Update tqdm postfix with losses
                tqdm.write(
                    f"[D Adv Loss: {avg_d_adv_loss:.4f}, D Grad Penalty: {avg_d_grad_penalty:.4f}, "
                    f"D Class Loss: {avg_d_class_loss:.4f}, D BPM Loss: {avg_d_bpm_loss:.4f}]"
                )

                # Log discriminator losses to TensorBoard.
                writer.add_scalar('Loss/Discriminator/Total', avg_d_loss, iteration)
                writer.add_scalar('Loss/Discriminator/Adversarial', avg_d_adv_loss, iteration)
                writer.add_scalar('Loss/Discriminator/GradPenalty', avg_d_grad_penalty, iteration)
                writer.add_scalar('Loss/Discriminator/Class', avg_d_class_loss, iteration)
                writer.add_scalar('Loss/Discriminator/BPM', avg_d_bpm_loss, iteration)

                # Log generator losses to TensorBoard.
                writer.add_scalar('Loss/Generator/Total', g_loss.item(), iteration)
                writer.add_scalar('Loss/Generator/Adversarial', g_adv_loss.item(), iteration)
                writer.add_scalar('Loss/Generator/Class', g_class_loss.item(), iteration)
                writer.add_scalar('Loss/Generator/BPM', g_bpm_loss.item(), iteration)

        # Save model weights at the end of each epoch.
        generator_path = os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.pth")
        discriminator_path = os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch+1}.pth")
        torch.save(generator.state_dict(), generator_path)
        torch.save(discriminator.state_dict(), discriminator_path)
        logger.info(f"Saved checkpoints for epoch {epoch+1} at '{checkpoint_dir}'")

    writer.close()

# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Create generator and discriminator instances.
    # Note: The generator's input_dim is now noise_dim + genre_dim + bpm_dim.
    generator = Generator(input_dim=NOISE_DIM + GENRE_DIM + BPM_DIM, output_dim=IMG_DIM, img_shape=(IMG_ROWS, IMG_COLS))
    discriminator = Discriminator(input_dim=IMG_DIM, num_genres=GENRE_DIM)

    dataset_directory = "bar_arrays"
    logger.info("Loading dataset")
    # Ensure that GrooveDataset returns a dictionary with "data" and "metadata" that also includes "bpm".
    dataset = GrooveDataset(root_dir=dataset_directory)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    logger.info("Starting up tensorboard")
    try:
        subprocess.Popen(["tensorboard", "--logdir=./runs/gdrm_experiment"])
        if sys.platform == "win32":
            subprocess.Popen(["start", "http://localhost:6006/"], shell=True)
        elif sys.platform == "linux":
            subprocess.Popen(["xdg-open", "http://localhost:6006/"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "http://localhost:6006/"])
        else:
            logger.info("Unsupported operating system")

    except Exception as e:
        print(f"{e}: \nCouldn't start tensorboard ")
    logger.info("Begin training")
    train(generator, discriminator, dataloader)
