import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ViTFeatureExtractor, ViTModel
import json
import torch.nn as nn
import torch.nn.functional as F
import csv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
class MultiFileDataset(Dataset):
    def __init__(self, image_npz_files, label_file, train=True):
        self.image_data = []
        self.labels = []

        # Load image data from npz files
        for npz_file in image_npz_files:
            data = np.load(npz_file, allow_pickle=True)
            for key in data.files:
                self.image_data.extend(data[key])
        
        # Load labels from the single JSON file
        with open(label_file, 'r') as f:
            labels = json.load(f)
            if train:
                for img_id in labels["train"]:
                    self.labels.append(labels["train"][img_id]["disease"]["disease"])
            else:
                for img_id in labels["test"]:
                    self.labels.append(labels["test"][img_id]["disease"]["disease"])
        
        if len(self.image_data) != len(self.labels):
            raise ValueError("The number of images and labels do not match.")
        
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.float32)

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
        # Assuming input size is [batch_size, 3, 224, 1120]
        regions = [input[:, :, :, i:i+224].to(input.device) for i in range(0, 1120, 224)]
        
        extracted_regions = [self.visual_extractor(region) for region in regions]

        diseases = [self.dropout1(self.bn1(F.relu(self.linear(region)))) for region in extracted_regions]
        linear56 = [self.bn2(F.relu(self.linear2(disease))) for disease in diseases]
        combined = torch.cat(linear56, dim=1)
        combined = self.dropout2(self.bn3(F.relu(self.linear3(combined))))
        result = F.tanh(combined)

        return result

def calculate_metrics(predictions, labels, num_classes):
    metrics = { 'precision': [], 'recall': [], 'f1_score': [], 'accuracy': [] }

    # We treat the predictions as a multilabel problem (each class can have -1, 0, or 1)
    for i in range(num_classes):
        y_true = np.array([label[i] for label in labels])
        y_pred = np.array([pred[i] for pred in predictions])

        # Calculate metrics for each class (-1, 0, 1)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=[-1, 0, 1])
        accuracy = accuracy_score(y_true, y_pred)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1_score)
        metrics['accuracy'].append(accuracy)

    return metrics

def print_metrics(metrics, class_names):
    for i, class_name in enumerate(class_names):
        print(f"Metrics for {class_name}:")
        print(f"  Precision: {metrics['precision'][i]:.4f}")
        print(f"  Recall: {metrics['recall'][i]:.4f}")
        print(f"  F1 Score: {metrics['f1_score'][i]:.4f}")
        print(f"  Accuracy: {metrics['accuracy'][i]:.4f}")
        print()

def evaluate_model(test_image_file, label_file, model_path, batch_size=16, device=None):
    # Load the pre-trained ViT model and feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Initialize the model and load the trained weights
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = diseaseDetector(vit_model, feature_extractor).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the test dataset
    test_dataset = MultiFileDataset(test_image_file, label_file, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    # Make predictions on the test dataset
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (outputs > 0).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Save predictions to a CSV file
    csv_file = "predictions_only_test.csv"
    diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", 
                "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
    metrics = calculate_metrics(all_preds, all_labels, num_classes=len(diseases))

    # Print metrics for each disease
    print_metrics(metrics, diseases)
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Pred ' + disease for disease in diseases]  # Only predictions
        writer.writerow(header)
        for i in range(len(all_preds)):
            row = list(all_preds[i])
            writer.writerow(row)

    print(f"Predictions saved to {csv_file}")

# Example usage:
test_image_file = ["/group/pmc023/rnandiya/test_data2.npz"]
label_file = 'final_data.json'
model_path = '/group/pmc023/rnandiya/model/disease_detector_model12.pth'
evaluate_model(test_image_file, label_file, model_path, batch_size=1000)
