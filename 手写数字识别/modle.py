
import torch
import torch.utils.data
from torch import nn


class FewShot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, padding="same"),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, padding="same"),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, 1, padding="same"),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 5, 1, padding="same"),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(256 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x