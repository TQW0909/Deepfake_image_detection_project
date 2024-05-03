import argparse
import copy
import sys
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchvision import models

# torch.use_deterministic_algorithms(True)

# # Load the pre-trained VGG-16 model and keeping the pre-trained weights
# vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# # Freeze the Convolutional Layers 
# for param in vgg16.features.parameters():
#     param.requires_grad = False

# # Example for unfreezing the last two conv blocks in VGG16:
# for layer in vgg16.features[-10:]:  # Adjust based on the structure of VGG-16
#     layer.requires_grad = True

# num_features = vgg16.classifier[0].in_features  # Get the input features of the classifier
# num_classes = 1  # For deepfake detection: real or fake
# vgg16.classifier = nn.Sequential(
#     nn.Linear(num_features, 512),  # Adding a new layer
#     nn.ReLU(True),
#     nn.Dropout(0.5),  # Dropout for regularization
#     nn.Linear(512, 128),  # Another new layer
#     nn.ReLU(True),
#     nn.Dropout(0.5),
#     nn.Linear(128, num_classes)  # Final layer for binary classification
# )

def create_vgg16_model(num_classes=1):

    # Load the pre-trained vgg-16 model
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Freeze all convolutional layers 
    for param in vgg16.features.parameters():
        param.requires_grad = False

    # Unfreeze the last 10 layers
    for layer in vgg16.features[-10:]:
        layer.requires_grad = True

    # Get input features of the classifier
    num_features = vgg16.classifier[0].in_features

    # Adding additional layers to original vgg model
    vgg16.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes) # Final output layers
    )

    return vgg16

