import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim=15, output_dim=96):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, output_dim)
        )

    def forward(self, noise, style, bpm):
        # Concatenate noise, style, and bpm to form the 15-dim input
        x = torch.cat((noise, style, bpm), dim=1)
        gen_output = self.model(x)
        # Append style and bpm to the generated output, resulting in a 106-dim vector
        output = torch.cat((gen_output, style, bpm), dim=1)
        return output
