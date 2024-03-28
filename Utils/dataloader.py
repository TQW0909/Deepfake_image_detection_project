import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from Utils.augmentation import *

DIRECTORY = '/Users/tingqiwang/Documents/USC/Spring 2024/CSCI 467/Homework/Project/Practical_Datasets/small_deepfake_and_real_images_subset/'

# Adjust the paths based on your dataset structure
train_data = datasets.ImageFolder(root=DIRECTORY + 'Train', 
                                  transform=deepfake_and_real_images_train_transforms)
dev_data = datasets.ImageFolder(root=DIRECTORY + 'Validation',
                                 transform=deepfake_and_real_images_val_test_transforms)
test_data = datasets.ImageFolder(root=DIRECTORY + 'Test',
                                 transform=deepfake_and_real_images_val_test_transforms)

# # Then ensure all samples use this new mapping
# train_data.samples = [(s, 1 - idx) for s, idx in train_data.samples]
# dev_data.samples = [(s, 1 - idx) for s, idx in train_data.samples]
# test_data.samples = [(s, 1 - idx) for s, idx in train_data.samples]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# batch_size = 32  # Adjust based on your GPU memory
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

def create_data_loader(data, batch_size, shuffle):
    """
    Create a DataLoader for the given data.
    
    :param data: Dataset object.
    :param batch_size: Size of each batch.
    :param shuffle: Whether to shuffle the data.
    :return: DataLoader with the specified batch size and shuffle settings.
    """
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
