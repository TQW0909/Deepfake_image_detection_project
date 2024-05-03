import pandas as pd
import os
import shutil
from tqdm import tqdm

source_dir = "../Datasets/deepfake_faces/faces_224"  # Update to path of the actual location for deepfake_faces dataset
subset_dir = "../Practical_Datasets/deepfake_faces_test"  # Update to path of the desired location for subset

# Load the csv into a df
df = pd.read_csv("../Datasets/deepfake_faces/metadata.csv") # Update to location of the 'metadata.vsc' of the deepfake_faces dataset in workspace

# Check that the image names and labels are present
assert 'videoname' in df.columns, "CSV must contain an 'videoname' column."
assert 'label' in df.columns, "CSV must contain a 'label' column."

# Modify the 'videoname' column to replace extensions with .jpg (was .mp4)
df['videoname'] = df['videoname'].apply(lambda x: os.path.splitext(x)[0] + ".jpg")

# Filter for a specific label:
df_sub = df[df['label'].isin(['FAKE', 'REAL'])]

# Create testset with 2500 samples per class (Change number as needed)
df_sub = df_sub.groupby('label').apply(lambda x: x.sample(min(len(x), 2500), random_state=42)).reset_index(drop=True)

# Create directories for each label
os.makedirs(subset_dir, exist_ok=True)
for label in df_sub['label'].unique():
    os.makedirs(os.path.join(subset_dir, label), exist_ok=True)

for _, row in tqdm(df_sub.iterrows()):
    image_name = row['videoname']
    label = row['label']

    src_path = os.path.join(source_dir, image_name)
    dst_path = os.path.join(subset_dir, label, image_name)

    # Check if the source image exists
    if os.path.isfile(src_path):
        # Copy or link the file
        shutil.copy2(src_path, dst_path)  # Copy the image



