import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim=1960, num_genres=18):
        """
        input_dim: flattened size of the sample data (e.g., 10*196 = 1960)
        num_genres: number of possible genres for the auxiliary classification task.
        """
        super(Discriminator, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Output for adversarial (real/fake) decision
        self.adv_layer = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        # Output for auxiliary classification (genre prediction)
        self.aux_layer = nn.Linear(256, num_genres)
        # Note: CrossEntropyLoss expects raw logits so we don’t apply any activation here.

    def forward(self, x):
        # x is expected to be (batch, input_dim) where input_dim = 10*196
        shared_out = self.shared(x)
        validity = self.adv_layer(shared_out)
        genre_logits = self.aux_layer(shared_out)
        return validity, genre_logits
