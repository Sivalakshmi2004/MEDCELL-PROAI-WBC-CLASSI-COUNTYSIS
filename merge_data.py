import os
import cv2 # type: ignore
import numpy as np
import shutil
from tqdm import tqdm # type: ignore
from glob import glob

# Define paths
source_datasets = {
    "Blood_Cells": r"C:\Users\Administrator\Desktop\proj\Train_wbc",  # Update with actual path
    "White_Blood_Cells": r"C:\Users\Administrator\Desktop\proj\TRAIN"  # Update with actual path
}

output_dir = "WBC_Dataset"
os.makedirs(output_dir, exist_ok=True)

# Define WBC categories
categories = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
for cat in categories:
    os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

# Resize parameters
IMG_SIZE = (224, 224)  # Standardizing image size

def preprocess_and_copy(source_path, category):
    """Preprocess images: resize and copy to target folder."""
    images = glob(os.path.join(source_path, "*.jpg")) + glob(os.path.join(source_path, "*.png"))
    target_folder = os.path.join(output_dir, category)
    
    for img_path in tqdm(images, desc=f"Processing {category}"):
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip corrupted images
        img = cv2.resize(img, IMG_SIZE)  # Resize
        img_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(target_folder, img_name), img)  # Save preprocessed image

# Process each dataset
for dataset_name, dataset_path in source_datasets.items():
    for category in categories:
        cat_path = os.path.join(dataset_path, category)
        if os.path.exists(cat_path):
            preprocess_and_copy(cat_path, category)

print("Dataset preprocessing and merging complete!")
