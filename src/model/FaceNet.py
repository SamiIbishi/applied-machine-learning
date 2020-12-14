# Torch Packages
from torch import nn
import torch.nn.functional as F


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Default: Input shape [NxCx512x512] => Output shape [Nx16x30x30]
        # [Input] => 4x[Conv2d => MaxPool2d => PReLU] => [Output]
        self.image_embedding = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(num_parameters=128, init=0.3),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(num_parameters=64, init=0.3),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(num_parameters=32, init=0.3),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(num_parameters=16, init=0.3),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(num_parameters=8, init=0.3),
        )

        self.image_vector = nn.Sequential(
            nn.Linear(1568, 512),
            nn.PReLU(num_parameters=1, init=0.3),
            nn.Linear(512, 256),
        )

    def forward_single(self, x):
        # Image Embedding
        x = self.image_embedding(x)
        x = x.view(-1, num_flat_features(x))
        x = self.image_vector(x)

        return x

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_single(anchor)
        positive_output = self.forward_single(positive)
        negative_output = self.forward_single(negative)
        return anchor_output, positive_output, negative_output
