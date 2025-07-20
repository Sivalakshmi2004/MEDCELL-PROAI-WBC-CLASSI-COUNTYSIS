from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import uvicorn
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import cv2
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 5)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

class_labels = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_wbc(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h
    return None

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)
    box = detect_wbc(image_np)

    if box:
        x, y, w, h = box
        cropped = image_np[y:y+h, x:x+w]
        cropped = cv2.resize(cropped, (224, 224))
    else:
        cropped = cv2.resize(image_np, (224, 224))

    cropped_pil = Image.fromarray(cropped)
    tensor_image = transform(cropped_pil).unsqueeze(0)

    return tensor_image, cropped, box

def generate_confusion_matrix_image(conf_matrix, class_names):
    plt.figure(figsize=(6, 5))
    plt.style.use("dark_background")

    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        square=True
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, color='cyan')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor='#121212')
    buf.seek(0)
    img_bytes = buf.getvalue()
    buf.close()
    plt.close()

    return base64.b64encode(img_bytes).decode("utf-8")

@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)):
    results = []
    y_true = []
    y_pred = []

    for file in files:
        contents = await file.read()
        img_tensor, cropped_np, box = preprocess_image(contents)

        # Debug: Check the shape and content of the image before prediction
        print("Image shape:", img_tensor.shape)
        print("Sample image pixel values:", img_tensor[0, 0, :10])  # Print a small sample of pixel values

        # Try to extract true label from filename
        filename = file.filename
        true_label = next((label for label in class_labels if label.lower() in filename.lower()), None)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = class_labels[predicted.item()]
            confidence_percentage = f"{confidence.item() * 100:.2f}%"

            # Debug: Check the raw output and probabilities
            print("Raw output:", outputs)
            print("Predicted class:", predicted_class)
            print("Confidence:", confidence_percentage)

            if true_label:
                y_true.append(true_label)
                y_pred.append(predicted_class)

            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            results.append({
                "filename": filename,
                "prediction": predicted_class,
                "confidence": confidence_percentage,
                "cropped_image": img_base64
            })

    # Evaluation
    if y_pred and y_true:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        confusion_img_base64 = generate_confusion_matrix_image(cm, class_labels)

        # Generate per-class metrics
        report = classification_report(y_true, y_pred, labels=class_labels, output_dict=True, zero_division=0)
        per_class_metrics = {
            label: {
                "precision": round(report[label]["precision"], 2),
                "recall": round(report[label]["recall"], 2),
                "f1_score": round(report[label]["f1-score"], 2),
                "support": report[label]["support"]
            }
            for label in class_labels
        }
    else:
        accuracy = precision = recall = f1 = 0.0
        confusion_img_base64 = ""
        per_class_metrics = {}

    return JSONResponse(content={
        "predictions": results,
        "evaluation_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_img_base64
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
