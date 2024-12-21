import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel
import json
import matplotlib.pyplot as plt


class SecondStageDataset(Dataset):
    def __init__(self, image_npz_file, first_stage_pred_npz_file, second_stage_pred_npz_file, label_file):
        # Load image data, ensure it is numeric and has the correct format
        image_data = np.load(image_npz_file, allow_pickle=True)
        self.image_data = np.array(image_data[image_data.files[0]], dtype=np.float32)  # Force conversion to float32
        print("Loaded image data")
        
        # Load prediction data and labels similarly
        first_stage_data = np.load(first_stage_pred_npz_file, allow_pickle=True)
        self.first_stage_predictions = np.array(first_stage_data[first_stage_data.files[0]], dtype=np.float32)
        print("Loaded first stage predictions")
        
        second_stage_data = np.load(second_stage_pred_npz_file, allow_pickle=True)
        self.second_stage_predictions = np.array(second_stage_data[second_stage_data.files[0]], dtype=np.float32)
        print("Loaded second stage predictions")
        
        label_data = np.load(label_file, allow_pickle=True)
        self.labels = np.array(label_data[label_data.files[0]], dtype=np.float32)
        print("Loaded label data")
        
        # Ensure all data files have the same length
        if len(self.image_data) != len(self.first_stage_predictions) or len(self.image_data) != len(self.second_stage_predictions):
            raise ValueError("The number of images, first stage predictions, second stage predictions, and labels do not match.")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]  # Full image (224x448)
        first_stage_pred = torch.tensor(self.first_stage_predictions[idx], dtype=torch.float32)
        second_stage_pred = torch.tensor(self.second_stage_predictions[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, first_stage_pred, second_stage_pred, label

class DiseaseDetectorSecondStage(nn.Module):
    def __init__(self, model, image_processor):
        super(DiseaseDetectorSecondStage, self).__init__()
        self.model = model
        self.image_processor = image_processor
        self.fc1 = nn.Linear(768*2 , 512)  # 768*2 for both image patches and 28 for the prediction vectors
        self.fc2 = nn.Linear(512 + 28, 256)
        self.fc3 = nn.Linear(256, 128)  # Output 3 classes for classification
        self.fc4 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(p=0.5)

        # Freeze the Vision Transformer parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def visual_extractor(self, region):
        # Process each region through the ViT
        inputs = self.image_processor(images=region, return_tensors="pt").to(region.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Extract the [CLS] token embedding

    def forward(self, input, first_stage_pred, second_stage_pred):
        # Assume input shape: [batch_size, 224, 448, 3]
        # Break down the image into overlapping patches of size 224x224
        regions = [input[:, :, :, i:i + 224].to(input.device) for i in range(0, 448, 224)]

        # Apply visual extractor on each patch
        extracted_regions = [self.visual_extractor(region) for region in regions]
        
        # Concatenate the features extracted from the patches
        combined_image_features = torch.cat(extracted_regions, dim=-1)  # [batch_size, 768*2]

        # Feed through dense layers
        x = F.relu(self.fc1(combined_image_features))
        x = self.dropout(x)
        x = torch.cat([x, first_stage_pred, second_stage_pred], dim=-1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Final output layer
        x = F.relu(self.fc4(x))

        return torch.sigmoid(x)  # Apply sigmoid for multi-label output

# Add accuracy calculation
def calculate_accuracy(outputs, labels):
    # Since labels are multi-label, we need to threshold at 0.5 for each output
    preds = (torch.sigmoid(outputs) > 0.6).float()  # Binarize the output
    correct = (preds == labels).float().sum()  # Count correct predictions
    return correct / labels.numel()  # Return accuracy as a fraction of total predictions
def calculate_metrics_manual(labels, preds):
    # Initialize true positives, false positives, false negatives
    tp = np.zeros(3)
    fp = np.zeros(3)
    fn = np.zeros(3)
    tn = np.zeros(3)

    for i in range(3):  # For each class: better, stable, worse
        for j in range(len(labels)):
            if preds[j][i] == 1 and labels[j][i] == 1:
                tp[i] += 1  # True Positive
            elif preds[j][i] == 1 and labels[j][i] == 0:
                fp[i] += 1  # False Positive
            elif preds[j][i] == 0 and labels[j][i] == 1:
                fn[i] += 1  # False Negative
            elif preds[j][i] == 0 and labels[j][i] == 0:
                tn[i] += 1  # True Negative

    # Calculate precision, recall, and F1-score for each class
    precision = np.zeros(3)
    recall = np.zeros(3)
    f1_score = np.zeros(3)

    for i in range(3):
        precision[i] = tp[i] / (tp[i] + fp[i] + 1e-10)  # Add small value to avoid division by zero
        recall[i] = tp[i] / (tp[i] + fn[i] + 1e-10)
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-10)

    # Calculate overall accuracy
    accuracy = (tp.sum() + tn.sum()) / (tp.sum() + tn.sum() + fp.sum() + fn.sum())

    return precision, recall, f1_score, accuracy

def predict_test_dataset(test_image_files, test_first_stage_pred_files, test_second_stage_pred_files, label_test_file, model_path, batch_size=500, device=None):
    # Load the pre-trained ViT model and feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Initialize the model and load the trained weights
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiseaseDetectorSecondStage(vit_model, feature_extractor).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Load test dataset
    test_dataset = SecondStageDataset(test_image_files, test_first_stage_pred_files, test_second_stage_pred_files, label_test_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Variables to store results
    all_labels = []
    all_preds = []

    # No gradient calculation for inference
    with torch.no_grad():
        for images, first_stage_preds, second_stage_preds, labels in test_loader:
            images, first_stage_preds, second_stage_preds, labels = images.to(device), first_stage_preds.to(device), second_stage_preds.to(device), labels.to(device)

            # Get predictions
            outputs = model(images, first_stage_preds, second_stage_preds)
            preds = (torch.sigmoid(outputs) > 0.7).float()  # Threshold at 0.6

            # Collect all labels and predictions
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    # Concatenate all predictions and labels across batches
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()

    # Calculate precision, recall, F1-score, and accuracy manually
    precision, recall, f1_score, accuracy = calculate_metrics_manual(all_labels, all_preds)

    # Output metrics
    class_names = ['Better', 'Stable', 'Worse']
    print(f"Overall Test Accuracy: {accuracy:.4f}")
    for i, class_name in enumerate(class_names):
        print(f"Class '{class_name}':")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1 Score: {f1_score[i]:.4f}")

test_image_files = '/group/pmc023/rnandiya/test_s2_image.npz'
test_first_stage_pred_files = '/group/pmc023/rnandiya/test_s2_pred_curr.npz'
test_second_stage_pred_files = '/group/pmc023/rnandiya/test_match_pred.npz'
label_test_file = '/group/pmc023/rnandiya/test_s2_label.npz'
model_path = '/group/pmc023/rnandiya/disease_detector_second_stage_model.pth'

predict_test_dataset(test_image_files, test_first_stage_pred_files, test_second_stage_pred_files, label_test_file, model_path)
