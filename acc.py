import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.efficientnet_b0(pretrained=False)
num_classes = 5  # Change if needed
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Define transforms and load test data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
data_dir = "WBC_DATASET"
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize tracking variables
class_correct = [0] * num_classes
class_total = [0] * num_classes
class_names = test_dataset.classes

# Calculate correct predictions per class
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for label, prediction in zip(labels, predicted):
            class_total[label.item()] += 1
            if label == prediction:
                class_correct[label.item()] += 1

# Compute per-class accuracy
class_accuracies = [100 * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]

# Plot per-class accuracy
colors = np.random.rand(len(class_names), 3)
plt.figure(figsize=(10, 6))
plt.bar(class_names, class_accuracies, color=colors)
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Per-Class Accuracy')
plt.ylim([0, 100])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
