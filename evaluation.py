from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def evaluate_model(y_true, y_pred):
    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    return precision, recall, f1, cm
               