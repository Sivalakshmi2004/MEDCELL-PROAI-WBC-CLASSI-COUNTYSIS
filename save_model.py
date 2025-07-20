import torch
import torchvision.models as models # type: ignore

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture (example: EfficientNet_B0)
model = models.efficientnet_b0(weights=None)

# Modify classifier if needed (based on your project!)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)  # Example: 4 classes
model = model.to(device)

torch.save(model.state_dict(), 'model.pth')  # Save the model weights
print("Model saved successfully!")
