import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim=24, output_dim=1920, img_shape=(10,192)):
        """
        input_dim: noise_dim + genre_dim + bpm_dim (e.g. 5 + 18 + 1 = 24)
        output_dim: flattened output size (e.g. 10*192 = 1920)
        img_shape: shape to reshape the output (10,192)
        """
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Ensures outputs between -1 and 1
        )

    def forward(self, noise, genre, bpm):
        # noise: (batch, noise_dim)
        # genre: (batch, genre_dim) -- one-hot vector
        # bpm: (batch, 1) - continuous value (make sure it is properly normalized)
        x = torch.cat((noise, genre, bpm), dim=1)  # (batch, noise_dim + genre_dim + 1)
        out = self.model(x)  # (batch, output_dim)
        # Reshape output to proper 2D shape per sample.
        output = out.view(out.size(0), *self.img_shape)
        return output
