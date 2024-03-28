import os
import numpy as np
import shutil
from tqdm import tqdm  # Optional, for the progress bar

def create_balanced_subset(source_dir, target_dir, num_images_per_category):
    """
    Create a balanced subset of data from the source directory and copy it into the target directory.
    
    :param source_dir: The directory of the original dataset
    :param target_dir: The directory where the balanced subset will be stored
    :param num_images_per_category: Number of images to copy for each category (Fake, Real)
    """
    # Make sure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over each category directory in the source directory (Fake, Real)
    for category_name in ['Fake', 'Real']:  # Explicitly specify categories to ensure balance
        category_dir = os.path.join(source_dir, category_name)
        if os.path.isdir(category_dir):
            # List all image files in the category directory
            images = [img for img in os.listdir(category_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            np.random.shuffle(images)  # Shuffle the list of images
            
            # Create a corresponding directory in the target directory
            target_category_dir = os.path.join(target_dir, category_name)
            if not os.path.exists(target_category_dir):
                os.makedirs(target_category_dir)
            
            # Copy the specified number of images to the target directory
            for img in tqdm(images[:num_images_per_category], desc=f'Copying {category_name}'):  # Remove tqdm if you don't want the progress bar
                src_path = os.path.join(category_dir, img)
                dst_path = os.path.join(target_category_dir, img)
                shutil.copy2(src_path, dst_path)

# Define your paths
original_dataset_dir = '../Datasets/deepfake_and_real_images'
subset_dir = '../Practical_Datasets/small_deepfake_and_real_images_subset'  # Change this to your desired subset directory path

# Specify the number of images per category (Fake, Real)
num_images_train = 350             # = 7000 / 2 Total images per category for training
num_images_validation = 100        # = 2000 / 2 Total images per category for validation
num_images_test = 50               # = 1000 / 2 Total images per category for test

# Create balanced training subset
create_balanced_subset(os.path.join(original_dataset_dir, 'Train'), os.path.join(subset_dir, 'Train'), num_images_train)

# Create balanced validation subset
create_balanced_subset(os.path.join(original_dataset_dir, 'Validation'), os.path.join(subset_dir, 'Validation'), num_images_validation)

# Create balanced test subset
create_balanced_subset(os.path.join(original_dataset_dir, 'Test'), os.path.join(subset_dir, 'Test'), num_images_validation)
