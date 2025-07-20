import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.efficientnet_b0(weights=None)
num_classes = 5
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Prepare test data
data_dir = "Dataset_Balanced"
test_dir = os.path.join(data_dir, "test")
class_names = sorted(os.listdir(test_dir))  # Ensure consistent order
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize matrix to store summed confidence scores
conf_matrix = np.zeros((num_classes, num_classes))  # [true][pred]
count_matrix = np.zeros((num_classes, num_classes))  # For averaging

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        for i in range(len(images)):
            true_label = labels[i].item()
            for pred_label in range(num_classes):
                conf_matrix[true_label][pred_label] += probs[i][pred_label].item()
                count_matrix[true_label][pred_label] += 1

# Avoid division by zero
avg_conf_matrix = np.divide(conf_matrix, count_matrix, out=np.zeros_like(conf_matrix), where=count_matrix!=0)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(avg_conf_matrix, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names, cmap="managua")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Average Confidence Scores per Class")
plt.tight_layout()
plt.show()
