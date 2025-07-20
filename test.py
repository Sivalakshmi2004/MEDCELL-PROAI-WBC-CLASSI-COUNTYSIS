import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the same model
model = models.efficientnet_b0(pretrained=False)  # Make sure you use the same variant!
num_classes = 5  # <-- Change this to your number of classes!
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(torch.load("model.pth", map_location=device))  # Load the trained model
model.to(device)
model.eval()

# Prepare test data
data_dir = "Dataset_Balanced"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_data = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Initialize lists to store predictions and true labels
all_preds = []
all_labels = []

# Testing
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # Append to lists
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
conf_matrix = confusion_matrix(all_labels, all_preds)

# Print metrics
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
