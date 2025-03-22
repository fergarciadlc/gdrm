import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    noise_dim = 5
    style_dim = 9
    bpm_dim = 1
    lr = 0.0002
    num_epochs = 100

    # Initialize models
    generator = Generator(input_dim=noise_dim + style_dim + bpm_dim, output_dim=96)
    discriminator = Discriminator(input_dim=96 + style_dim + bpm_dim)

    # Loss function - you might experiment with alternatives for stability
    adversarial_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Assume you have a dataloader that yields tuples: (real_vector, style, bpm)
    for epoch in range(num_epochs):
        for i, (real_data, style_data, bpm_data) in enumerate(dataloader):
            current_batch = real_data.size(0)

            # Adversarial ground truths
            valid = torch.ones(current_batch, 1)
            fake = torch.zeros(current_batch, 1)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            noise = torch.randn(current_batch, noise_dim)
            # For conditional GAN, style and bpm can be taken from the dataset or sampled;
            # here we assume they are provided by the dataloader
            gen_data = generator(noise, style_data, bpm_data)
            g_loss = adversarial_loss(discriminator(gen_data), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Loss for real data
            real_loss = adversarial_loss(discriminator(real_data), valid)
            # Loss for fake data
            fake_loss = adversarial_loss(discriminator(gen_data.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
