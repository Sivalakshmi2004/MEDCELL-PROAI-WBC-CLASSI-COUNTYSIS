import matplotlib.pyplot as plt
import torch
import pickle  # If you saved history as pickle
# If you saved history in a different format (e.g., JSON), use the corresponding library.

# Load history (this assumes you saved it as a dictionary using pickle, modify as needed)
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Plotting training and validation loss and accuracy
epochs = range(1, len(history['train_acc']) + 1)

plt.figure(figsize=(12, 6))

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_acc'], label='Train Accuracy')
plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, history['train_loss'], label='Train Loss')
plt.plot(epochs, history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
