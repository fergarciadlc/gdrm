"""
Training loop for a conditional GAN with auxiliary classifiers for genre and BPM.
Each sample is a 2D array of shape (10,196) with values in [-1, 1].
Conditions:
  - Genre: provided as a one-hot vector (dim 18)
  - BPM: a continuous value (normalized) with shape (1,)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

# The generated “drum pattern” is 10x196 = 1960 numbers.
IMG_ROWS = 10
IMG_COLS = 192
IMG_DIM = IMG_ROWS * IMG_COLS

# Loss functions:
# For adversarial loss we use BCE; for genre classification, CrossEntropy; for BPM regression, MSE.
adversarial_loss_fn = nn.BCELoss()
classification_loss_fn = nn.CrossEntropyLoss()
bpm_loss_fn = nn.MSELoss()

def train(generator: Generator, discriminator: Discriminator, dataloader):
    # Optionally, move models and loss functions to the GPU.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # generator.to(device)
    # discriminator.to(device)
    # adversarial_loss_fn.to(device)
    # classification_loss_fn.to(device)
    # bpm_loss_fn.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(tqdm(dataloader)):
            real_data = batch["data"]                     # shape: (batch, 10,192)
            genre_data = batch["metadata"]["genre"]         # shape: (batch, 18) one-hot
            bpm_data = batch["metadata"]["bpm"].unsqueeze(dim=1)  # shape: (batch, 1); ensure it is normalized

            current_batch = real_data.size(0)

            # Create ground truths for adversarial loss:
            valid = torch.ones(current_batch, 1)
            fake  = torch.zeros(current_batch, 1)

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
            real_adv_loss = adversarial_loss_fn(real_validity, valid)
            real_class_loss = classification_loss_fn(real_genre_pred, real_genre_labels)
            real_bpm_loss = bpm_loss_fn(real_bpm_pred, bpm_data)

            # Generate fake samples:
            noise = torch.randn(current_batch, NOISE_DIM)
            # Generator now requires noise, genre, and bpm conditioning.
            fake_data = generator(noise, genre_data, bpm_data)  # fake_data shape: (batch,10,192)
            fake_data_flat = fake_data.view(current_batch, -1)

            # Forward pass on fake samples (detach to avoid gradients flowing into generator)
            fake_validity, fake_genre_pred, fake_bpm_pred = discriminator(fake_data_flat.detach())
            fake_adv_loss = adversarial_loss_fn(fake_validity, fake)

            # Even on fake drum patterns, we want the auxiliary branches to predict the conditioning:
            fake_class_loss = classification_loss_fn(fake_genre_pred, real_genre_labels)
            fake_bpm_loss = bpm_loss_fn(fake_bpm_pred, bpm_data)

            # Total discriminator loss:
            d_loss = ((real_adv_loss + fake_adv_loss) * 0.5 +
                      (real_class_loss + fake_class_loss) * 0.5 +
                      (real_bpm_loss + fake_bpm_loss) * 0.5)
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Generate fake samples with the same conditioning:
            gen_data = generator(noise, genre_data, bpm_data)
            gen_data_flat = gen_data.view(current_batch, -1)
            validity, genre_pred, bpm_pred = discriminator(gen_data_flat)

            # Generator loss: try to fool the discriminator and also match conditioning.
            g_adv_loss = adversarial_loss_fn(validity, valid)
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
