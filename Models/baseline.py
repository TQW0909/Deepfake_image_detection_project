import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# torch.use_deterministic_algorithms(True)

IMAGE_SHAPE = (256, 256)  # Size of Deepfake and real images
NUM_CLASSES = 1  # Number of classes we are classifying over (FAKE | REAL)


class BaselineCNN(nn.Module):
    def __init__(self, num_out_channels=16, kernel_size=3, hidden_dim=256, dropout_prob=0.5):
        super(BaselineCNN, self).__init__()

        # Convolutional layer (sees 256x256x3 image tensor)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_out_channels, kernel_size=3, stride=1, padding=1)
        # Convolutional layer (sees 128x128x16 tensor)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Convolutional layer (sees 64x64x32 tensor)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Linear layer (64 * 32 * 32 -> hidden_dim)
        self.linear1 = nn.Linear(64 * 32 * 32, hidden_dim)
        # Linear layer (hidden_dim -> 1)
        self.linear2 = nn.Linear(hidden_dim, 1)
        # Dropout layer (p=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x):
        """Output the predicted scores for each class.

        The outputs are the scores *before* the softmax function.

        Inputs:
            x: Torch tensor of size (B, D)
        Outputs:
            Matrix of size (B, C).
        """

        conv1_output = self.pool(F.relu(self.conv1(x)))
        conv2_output = self.pool(F.relu(self.conv2(conv1_output)))
        conv3_output = self.pool(F.relu(self.conv3(conv2_output)))

        flatten_output = conv3_output.flatten(start_dim=1)

        dropout1_output = self.dropout(flatten_output)
        
        linear1_output = F.relu(self.linear1(dropout1_output))

        dropout2_output = self.dropout(linear1_output)

        output = self.linear2(dropout2_output)

        return output
    