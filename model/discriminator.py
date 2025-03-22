import torch
import torch.nn as nn

# Discriminator: takes a 106-dim vector and outputs a probability
class Discriminator(nn.Module):
    def __init__(self, input_dim=106):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # For standard GAN training; remove for WGAN variants
        )

    def forward(self, x):
        validity = self.model(x)
        return validity
