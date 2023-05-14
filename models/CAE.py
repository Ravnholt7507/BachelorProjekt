from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
          nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
          nn.ReLU(),
          nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
          nn.BatchNorm2d(24),
          nn.ReLU(),
          nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
          nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
          nn.ReLU(),
          nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
          nn.BatchNorm2d(12),
          nn.ReLU(),
          nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
          nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x