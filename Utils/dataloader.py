import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from Utils.augmentation import *

base_path = os.path.dirname(os.path.abspath("Project"))

DIRECTORY = os.path.join(base_path, 'Practical_Datasets', '50k_deepfake_and_real_images_subset/') # Change to path of deepfake_and_real_images subset

TEST_DIRECTORY = os.path.join(base_path, 'Practical_Datasets', 'deepfake_faces_test/') # Change to path of deepfake_faces subset

# Adjust the paths based on your dataset structure
train_data = datasets.ImageFolder(root=DIRECTORY + 'Train', 
                                  transform=deepfake_and_real_images_train_transforms)
dev_data = datasets.ImageFolder(root=DIRECTORY + 'Validation',
                                 transform=deepfake_and_real_images_val_test_transforms)
test_data = datasets.ImageFolder(root=DIRECTORY + 'Test',
                                 transform=deepfake_and_real_images_val_test_transforms)

deepfake_faces_test_data = datasets.ImageFolder(root=TEST_DIRECTORY,
                                 transform=deepfake_faces_val_test_transforms)



# train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
# dev_loader = DataLoader(dev_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
# test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

# deepfake_faces_test_loader = DataLoader(deepfake_faces_test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

deepfake_faces_test_loader = DataLoader(deepfake_faces_test_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

# Used for hyperparameter tuning
def create_data_loader(data, batch_size, shuffle):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, prefetch_factor=2)
