import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, input_dim=24, output_dim=1920, img_shape=(10, 192)):
        """
        input_dim: noise_dim + genre_dim + bpm_dim (e.g. 5 + 18 + 1 = 24)
        output_dim: flattened output size (e.g. 10*192 = 1920)
        img_shape: shape to reshape the output (10,192)
        """
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 128)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Linear(128, 256)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Linear(256, output_dim)),
            nn.Tanh()  # Ensures outputs between -1 and 1.
        )

    def realify_genered_pattern(self, fake_pattern: torch.Tensor):
        neg_one = torch.tensor(-1.0, device=fake_pattern.device)
        return torch.where(fake_pattern <= 0, neg_one, fake_pattern)

    def forward(self, noise, genre, bpm):
        # noise: (batch, noise_dim)
        # genre: (batch, genre_dim) -- one-hot vector
        # bpm: (batch, 1) - continuous value (make sure it is properly normalized).
        x = torch.cat((noise, genre, bpm), dim=1)  # (batch, noise_dim + genre_dim + 1)
        out = self.model(x)  # (batch, output_dim)
        # Reshape output to proper 2D shape per sample.
        output = out.view(out.size(0), *self.img_shape)
        pattern = self.realify_genered_pattern(output)
        return pattern
