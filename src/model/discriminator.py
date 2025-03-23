import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim=1960, num_genres=18):
        """
        input_dim: flattened size of the sample data (e.g., 10*196 = 1960)
        num_genres: number of genres for the auxiliary classification task.
        """
        super(Discriminator, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Output for adversarial (real/fake) decision - raw score for Wasserstein distance.
        self.adv_layer = nn.Linear(256, 1)
        # Auxiliary output for genre prediction
        self.aux_layer = nn.Linear(256, num_genres)
        # Auxiliary output for BPM regression – predicting a continuous value.
        self.bpm_layer = nn.Linear(256, 1)
        # Note: For genre classification (and BPM) no activation is used since we apply the proper loss functions.

    def forward(self, x):
        # x is expected to be (batch, input_dim) where input_dim = 10*196 (flattened)
        shared_out = self.shared(x)
        validity = self.adv_layer(shared_out)  # raw output for Wasserstein loss
        genre_logits = self.aux_layer(shared_out)
        bpm_pred = self.bpm_layer(shared_out)
        return validity, genre_logits, bpm_pred
