import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define transforms for the training, validation, and test sets for deepfake_faces dataset
deepfake_faces_train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet standards
])

deepfake_faces_val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define transforms for the training, validation, and test sets for deepfake_and_real_images dataset
deepfake_and_real_images_train_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(256),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet standards
])

deepfake_and_real_images_val_test_transforms = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
