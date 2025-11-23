import torch
import torch.nn as nn
from .residual_block import ResidualBlock, CBAM


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # GRAYSCALE INPUT
            nn.ReLU(),
            ResidualBlock(32),
            CBAM(32)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            CBAM(64)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            CBAM(128)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        return x1, x2, x3


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            CBAM(64)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(32),
            CBAM(32)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),   # GRAYSCALE OUTPUT
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3):
        d3 = self.dec3(x3)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        return d1


class AA_RAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        out = self.decoder(x1, x2, x3)
        return out
