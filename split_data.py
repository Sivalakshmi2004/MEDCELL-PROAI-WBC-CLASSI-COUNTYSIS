import os
import shutil
import random
from tqdm import tqdm

# Paths
source_dir = "Dataset_Balanced"
output_dir = "Dataset_Split"
train_ratio = 0.8  # 80% train, 20% test

# Create train/test directories
for split in ["train", "test"]:
    split_path = os.path.join(output_dir, split)
    os.makedirs(split_path, exist_ok=True)

# Process each class folder
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_idx = int(train_ratio * len(images))
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Create class subfolders
    for split_name, split_images in [("train", train_images), ("test", test_images)]:
        split_class_dir = os.path.join(output_dir, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for img_name in tqdm(split_images, desc=f"{split_name.capitalize()} {class_name}"):
            src = os.path.join(class_path, img_name)
            dst = os.path.join(split_class_dir, img_name)
            shutil.copyfile(src, dst)

print("✅ Dataset split complete! Saved to 'Dataset_Split'")
