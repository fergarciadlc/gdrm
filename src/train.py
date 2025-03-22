"""
Training loop for a conditional GAN with auxiliary classifier.
Data: each sample is a 2D array of shape (10,196) with values in [-1, 1].
Conditioning: a genre given as a one-hot vector (dim 18).
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
NUM_EPOCHS = 100
LEARNING_RATE = 0.0002

# The generated “image” is 10x196 = 1960 numbers.
IMG_ROWS = 10
IMG_COLS = 192
IMG_DIM = IMG_ROWS * IMG_COLS

# Loss functions: adversarial (BCE) and auxiliary classifier (CrossEntropy)
adversarial_loss_fn = nn.BCELoss()
classification_loss_fn = nn.CrossEntropyLoss()

def train(generator: Generator, discriminator: Discriminator, dataloader: DataLoader):
    # If desired: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # And then move generator, discriminator, and losses to device.

    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    for epoch in range(NUM_EPOCHS):
        for i,  batch in enumerate(tqdm(dataloader)):
            real_data = batch["data"]
            genre_data = batch["metadata"]["genre"]

            current_batch = real_data.size(0)

            # Create ground truths for adversarial loss
            valid = torch.ones(current_batch, 1)
            fake  = torch.zeros(current_batch, 1)

            # (Assume genre_data is one-hot. If needed, convert to label indices.)
            # For classification losses the CrossEntropyLoss expects target labels as LongTensor
            real_genre_labels = genre_data.argmax(dim=1)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Flatten real_data: shape (batch, 10,196) -> (batch,1960)
            real_data_flat = real_data.view(current_batch, -1)
            # Forward pass on real samples
            real_validity, real_genre_pred = discriminator(real_data_flat)
            real_adv_loss = adversarial_loss_fn(real_validity, valid)
            real_class_loss = classification_loss_fn(real_genre_pred, real_genre_labels)

            # Generate fake samples
            noise = torch.randn(current_batch, NOISE_DIM)
            # Generator takes noise & genre as conditioning. (No need to pass extra info after output.)
            fake_data = generator(noise, genre_data)  # fake_data shape: (batch,10,196)
            fake_data_flat = fake_data.view(current_batch, -1)

            # Forward pass on fake samples.
            fake_validity, fake_genre_pred = discriminator(fake_data_flat.detach())
            fake_adv_loss = adversarial_loss_fn(fake_validity, fake)
            # For fake images we want the classifier to “see” the conditioning that was fed to the generator.
            # In other words, the discriminator’s auxiliary classifier should predict the genre label correctly,
            # even for fake data.
            fake_class_loss = classification_loss_fn(fake_genre_pred, real_genre_labels)

            # Total discriminator loss: sum or average of adversarial + classification losses.
            d_loss = (real_adv_loss + fake_adv_loss) * 0.5 + (real_class_loss + fake_class_loss) * 0.5
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Generate fake images using the same noise and genre conditioning.
            gen_data = generator(noise, genre_data)
            gen_data_flat = gen_data.view(current_batch, -1)
            validity, genre_pred = discriminator(gen_data_flat)

            # Generator wants the discriminator to classify fake as real…
            g_adv_loss = adversarial_loss_fn(validity, valid)
            # …and also to "win" on the auxiliary classifier task
            g_class_loss = classification_loss_fn(genre_pred, real_genre_labels)
            g_loss = g_adv_loss + g_class_loss

            g_loss.backward()
            optimizer_G.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

if __name__ == "__main__":

    # Create generator and discriminator instances.
    generator = Generator(input_dim=NOISE_DIM + GENRE_DIM, output_dim=IMG_DIM, img_shape=(IMG_ROWS, IMG_COLS))
    discriminator = Discriminator(input_dim=IMG_DIM, num_genres=GENRE_DIM)

    dataset_directory = "bar_arrays"

    print("loading dataset")
    dataset = GrooveDataset(root_dir=dataset_directory)  # Make sure your dataset returns (data, genre)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("Begin training")
    train(generator, discriminator, dataloader)
