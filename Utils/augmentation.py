import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


deepfake_faces_val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet standard weights
])

# Augmentation for training set taken from deepfake_and_real_images
deepfake_and_real_images_train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Flip 50% of the time
    transforms.RandomVerticalFlip(p=0.5),  # Flip 50% of the time
    transforms.RandomRotation(degrees=30),  # Rotate images by up to 30 degrees in either direction
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet standard weights
])

deepfake_and_real_images_val_test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet standard weights
])
