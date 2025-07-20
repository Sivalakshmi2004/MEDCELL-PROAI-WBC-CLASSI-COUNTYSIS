import os
import shutil
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Paths
input_dir = "WBC_Dataset"
output_dir = "Dataset_Balanced"
os.makedirs(output_dir, exist_ok=True)

# Data augmentation transforms
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
])

# Step 1: Count images per class
class_counts = {}
max_count = 0
for cls in os.listdir(input_dir):
    cls_path = os.path.join(input_dir, cls)
    if os.path.isdir(cls_path):
        images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        class_counts[cls] = len(images)
        max_count = max(max_count, len(images))

# Step 2: Augment and copy to output directory
for cls, count in class_counts.items():
    cls_input_path = os.path.join(input_dir, cls)
    cls_output_path = os.path.join(output_dir, cls)
    os.makedirs(cls_output_path, exist_ok=True)

    images = [f for f in os.listdir(cls_input_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Copy existing images
    for img_name in images:
        src = os.path.join(cls_input_path, img_name)
        dst = os.path.join(cls_output_path, img_name)
        shutil.copy(src, dst)

    # Add augmented images to reach max_count
    idx = 0
    pbar = tqdm(total=max_count - count, desc=f"Augmenting {cls}")
    while len(os.listdir(cls_output_path)) < max_count:
        img_path = os.path.join(cls_input_path, images[idx % len(images)])
        with Image.open(img_path) as img:
            img = augment_transform(img)
            aug_name = f"aug_{idx}_{os.path.basename(img_path)}"
            img.save(os.path.join(cls_output_path, aug_name))
        idx += 1
        pbar.update(1)
    pbar.close()

print("✅ Dataset balancing complete. Output saved to 'WBC_Dataset_Balanced/train'")
