# Torch Packages
from torch import nn
import torch.nn.functional as F


class FaceNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        """

        :param in_channels:
        :param n_classes:
        """
        super(FaceNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=11, stride=4, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(24, 64, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, n_classes)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)

        x = self.pool3(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        y_pred = self.output_layer(x)

        return y_pred