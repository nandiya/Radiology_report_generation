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

def process_json_files(bbox_json_file, path_json_file, root_path, img_id):
    with open(bbox_json_file, 'r') as bbox_file:
        bbox_data = json.load(bbox_file)

    with open(path_json_file, 'r') as path_file:
        path_data = json.load(path_file)
    
    # Check if img_id exists in the path_data
    if img_id in path_data["train"]: 
        image_path = os.path.join(root_path, path_data["train"][img_id]["disease"]["image_path"][0])
        if os.path.exists(image_path):  
            bboxes = bbox_data[img_id]
            torch_image = process_image(image_path, bboxes["bboxes"])
            return torch_image  # Return the processed image tensor
    return None


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

def evaluate_model(image_tensor, label_file, model_path, img_id, device=None):
    # Load the pre-trained ViT model and feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Initialize the model and load the trained weights
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = diseaseDetector(vit_model, feature_extractor).to(device)
    
    # Load the model weights, mapping to CPU if necessary
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()

    # Ensure the image tensor is on the correct device
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Load the corresponding label for the img_id
    with open(label_file, 'r') as f:
        label_data = json.load(f)
        if img_id in label_data["train"]:
            label = label_data["train"][img_id]["disease"]["disease"]
            label_tensor = torch.tensor(label, dtype=torch.float32).to(device)
        else:
            print(f"Label for img_id {img_id} not found.")
            return

    # Make predictions and print them with labels
    with torch.no_grad():
        output = model(image_tensor)
        pred = (output > 0.5).float()

    # Print prediction and corresponding label
    print("Prediction and corresponding label:")
    print(f"Prediction: {pred.cpu().numpy()}, Label: {label_tensor.cpu().numpy()}")



bbox_json_file = "train_bb_4s.json"
path_json_file = "final_data.json"
root_path = "/group/pmc023/rnandiya/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
img_id = "2a280266-c8bae121-54d75383-cac046f4-ca37aa16"

# Process the image for the given img_id
image_tensor = process_json_files(bbox_json_file, path_json_file, root_path, img_id)

# Only proceed if image_tensor is not None
if image_tensor is not None:
    label_file = 'final_data.json'
    model_path = '/group/pmc023/rnandiya/model/disease_detector_model11.pth'
    evaluate_model(image_tensor, label_file, model_path, img_id)
else:
    print("Image could not be found or processed.")
