import json
import os
import numpy as np
from PIL import Image
import torch

import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ViTFeatureExtractor, ViTModel
import torch.nn as nn
import torch.nn.functional as F
import csv

# Functions for loading, processing, and concatenating images
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def resize_image(image, size=(448, 448)):
    return image.resize(size, Image.Resampling.LANCZOS)

def crop_and_resize(image, bboxes, crop_size=(224, 224)):
    cropped_images = []
    for (x, y, w, h) in bboxes:
        cropped_region = image.crop((x, y, x + w, y + h))
        resized_region = cropped_region.resize(crop_size, Image.Resampling.LANCZOS)
        cropped_images.append(resized_region)
    return cropped_images

def concatenate_images(images, grid_size=(1, 4)):
    rows = []
    for i in range(0, len(images), grid_size[1]):
        row_images = images[i:i + grid_size[1]]
        row = np.concatenate([np.array(img) for img in row_images], axis=1)
        rows.append(row)
    concatenated_image = np.concatenate(rows, axis=0)
    return concatenated_image

def process_image(image_path, bboxes):
    image = load_image(image_path)
    resized_image = resize_image(image, size=(448, 448))
    resized_whole_image = resize_image(image, size=(224, 224))
    cropped_images = [resized_whole_image] + crop_and_resize(resized_image, bboxes, crop_size=(224, 224))
    concatenated_image = concatenate_images(cropped_images, grid_size=(1, 5))
    torch_image = torch.tensor(concatenated_image, dtype=torch.float32).permute(2, 0, 1)
    return torch_image



# Model definition
class diseaseDetector(nn.Module):
    def __init__(self, model, image_processor):
        super(diseaseDetector, self).__init__()
        self.model = model
        self.image_processor = image_processor
        self.linear = nn.Linear(768, 96)
        self.bn1 = nn.BatchNorm1d(96)
        self.linear2 = nn.Linear(96, 14)
        self.bn2 = nn.BatchNorm1d(14)
        self.dropout1 = nn.Dropout(p=0.4)
        self.linear3 = nn.Linear(14 * 5, 14)
        self.bn3 = nn.BatchNorm1d(14)
        self.dropout2 = nn.Dropout(p=0.4)

        # Freeze the Vision Transformer parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def visual_extractor(self, region):
        inputs = self.image_processor(images=region, return_tensors="pt").to(region.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states[:, 0, :]

    def forward(self, input):
        regions = [input[:, :, :, i:i+224].to(input.device) for i in range(0, 1120, 224)]
        extracted_regions = [self.visual_extractor(region) for region in regions]
        diseases = [self.dropout1(self.bn1(F.relu(self.linear(region)))) for region in extracted_regions]
        linear56 = [self.bn2(F.relu(self.linear2(disease))) for disease in diseases]
        combined = torch.cat(linear56, dim=1)
        combined = self.dropout2(self.bn3(F.relu(self.linear3(combined))))
        return combined

# Function to calculate metrics
def calculate_metrics(predictions, labels, num_classes):
    metrics = { 'precision': [], 'recall': [], 'f1_score': [], 'accuracy': [] }
    for i in range(num_classes):
        y_true = np.array([label[i] for label in labels])
        y_pred = np.array([pred[i] for pred in predictions])
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        precision =  0 if (TP + FP) == 0 else TP / (TP + FP)
        recall = 0 if  (TP + FN) == 0 else TP / (TP + FN) 
        f1_score = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall) 
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1_score)
        metrics['accuracy'].append(accuracy)
    return metrics

# Function to print metrics
def print_metrics(metrics, class_names):
    for i, class_name in enumerate(class_names):
        print(f"Metrics for {class_name}:")
        print(f"  Precision: {metrics['precision'][i]:.4f}")
        print(f"  Recall: {metrics['recall'][i]:.4f}")
        print(f"  F1 Score: {metrics['f1_score'][i]:.4f}")
        print(f"  Accuracy: {metrics['accuracy'][i]:.4f}")
        print()

# Evaluation function
def evaluate_model(bbox_json_file, path_json_file, root_path, label_file, model_path, batch_size=16, device=None):
    # Load the pre-trained ViT model and feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Initialize the model and load the trained weights
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = diseaseDetector(vit_model, feature_extractor).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Process images from JSON files
    images = process_json_files(bbox_json_file, path_json_file, root_path)
    all_labels = []

    # Load labels from the JSON file
    with open(label_file, 'r') as f:
        labels_data = json.load(f)
        for img_id in labels_data["train"]:
            all_labels.append(labels_data["train"][img_id]["disease"]["disease"])

    # Ensure the labels match the processed images
    if len(images) != len(all_labels):
        raise ValueError("The number of images and labels do not match.")

    all_preds = []

    # Make predictions on the test dataset
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_images = torch.stack(batch_images).to(device)
            outputs = model(batch_images)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Save predictions and labels to a CSV file
    csv_file = "predictions_validation.csv"
    diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", 
                "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['True ' + disease for disease in diseases] + ['Pred ' + disease for disease in diseases]
        writer.writerow(header)
        for i in range(len(all_preds




bbox_json_file = "train_bb_4s.json"  

## change the json to the chunk file for training or final_data.json for val and testing
path_json_file = "final_data.json"  
root_path = "/group/pmc023/rnandiya/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"  
img_id="fffabebf-74fd3a1f-673b6b41-96ec0ac9-2ab69818"
result = process_json_files(bbox_json_file, path_json_file, root_path)
test_image_file = result
label_file = 'final_data.json'
model_path = '/group/pmc023/rnandiya/model/disease_detector_model11.pth'
evaluate_model(test_image_file, label_file, model_path, batch_size=1000)