import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset path
data_dir = "Dataset_Split"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load test dataset
test_data = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Rebuild model architecture and load weights
model = models.efficientnet_b0(weights=None)
num_classes = len(test_data.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

# Evaluate
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - WBC Test Set")
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved as confusion_matrix.png")

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=test_data.classes))
