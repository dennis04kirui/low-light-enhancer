import torch
import torch.nn as nn
import torch.nn.functional as F

class LowLightEnhancer(nn.Module):
    def __init__(self):
        super(LowLightEnhancer, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        # channel attention
        self.channel_att = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        # spatial attention
        self.spatial_att = nn.Conv2d(64, 1, 1)
        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        enc = self.encoder(x)
        # simplified attention (for demo)
        x = self.decoder(enc)
        return torch.sigmoid(x)
