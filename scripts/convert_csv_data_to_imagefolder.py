import os
import shutil
import pandas as pd
from pathlib import Path

# Define paths
base_dir = Path("untracked/cassava-leaf-disease-classification")
test_images_dir = Path("untracked/cassava-leaf-disease-classification/test_images")
train_images_dir = Path("untracked/cassava-leaf-disease-classification/train_images")   # Directory containing all images
csv_file = "untracked/cassava-leaf-disease-classification/train.csv"  # CSV file with image_id, label, image_name

# Read CSV file
df = pd.read_csv(csv_file)

# Map labels to folder names (optional customization)
label_map = {0: "Cassava Bacterial Blight (CBB)", 1: "Cassava Brown Streak Disease (CBSD)", 2: "Cassava Green Mottle (CGM)", 3: "Cassava Mosaic Disease (CMD)", 4: "Healthy"}


# Create directory structure
for label in label_map.values():
    (base_dir / label).mkdir(parents=True, exist_ok=True)

# Move images to respective label folders
for _, row in df.iterrows():
    image_name = row['image_id']  # e.g., '1000015157.jpg'
    label = row['label']  # e.g., 0
    src_path = train_images_dir / image_name
    dst_path = base_dir / label_map[label] / image_name
    if src_path.exists():  # Check if image exists
        shutil.copy(src_path, dst_path)  # Copy to preserve original
        print(f"Copied {image_name} to {label_map[label]}")
    else:
        print(f"Warning: {image_name} not found in {train_images_dir}")

print("Conversion complete. Dataset ready at:", base_dir)