import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import ViTFeatureExtractor, ViTModel
import json
import matplotlib.pyplot as plt



class SecondStageDataset(Dataset):
    def __init__(self, image_npz_files, label_file, first_stage_predictions_csv, train=True):
        self.image_data = []
        self.labels = []
        self.first_stage_predictions = []

        # Load image data from npz files
        for npz_file in image_npz_files:
            data = np.load(npz_file, allow_pickle=True)
            for key in data.files:
                self.image_data.extend(data[key])
        
        # Load labels from the JSON file
        with open(label_file, 'r') as f:
            labels = json.load(f)
            if train:
                for img_id in labels["train"]:
                    self.labels.append(labels["train"][img_id]["progress"]["disease"])
            else:
                for img_id in labels["val"]:
                    self.labels.append(labels["val"][img_id]["disease"]["disease"])
        
        # Load first-stage predictions and extract all "Pred" columns
        predictions_df = pd.read_csv(first_stage_predictions_csv)

        # Selecting all "Pred" columns
        self.first_stage_predictions = predictions_df.filter(like="Pred").values

        # Check if the number of images matches the number of labels and predictions
        if len(self.image_data) != len(self.first_stage_predictions):
            raise ValueError("The number of images, labels, and predictions do not match.")
        
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        first_stage_pred = torch.tensor(self.first_stage_predictions[idx], dtype=torch.float32)
        return image, label, first_stage_pred


class DiseaseDetectorSecondStage(nn.Module):
    def __init__(self, model, image_processor):
        super(DiseaseDetectorSecondStage, self).__init__()
        self.model = model
        self.image_processor = image_processor
        self.linear = nn.Linear(768 + 14, 96)  # Adjusting the input size to include 14 additional features
        self.bn1 = nn.BatchNorm1d(96)
        self.linear2 = nn.Linear(96, 3)  # Adjust output to 3 classes for second stage
        self.bn2 = nn.BatchNorm1d(3)
        self.dropout1 = nn.Dropout(p=0.4)

        # Freeze the Vision Transformer parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def visual_extractor(self, region):
        inputs = self.image_processor(images=region, return_tensors="pt").to(region.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states[:, 0, :]

    def forward(self, input, first_stage_pred):
        regions = [input[:, :, :, i:i+224].to(input.device) for i in range(0, 1120, 224)]
        extracted_regions = [self.visual_extractor(region) for region in regions]

        # Concatenate first stage predictions to the extracted features
        combined_features = torch.cat(extracted_regions + [first_stage_pred.unsqueeze(1)], dim=-1)

        diseases = self.dropout1(self.bn1(F.relu(self.linear(combined_features))))
        result = self.bn2(F.relu(self.linear2(diseases)))

        return result

def train_second_stage_model(train_image_files, label_file, val_image_file, 
                             train_first_stage_predictions_csv, val_first_stage_predictions_csv,
                             num_epochs=1, batch_size=16, learning_rate=1e-4, 
                             checkpoint_path=None, device=None):
    # Load the pre-trained ViT model and feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Initialize the model, loss function, and optimizer
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiseaseDetectorSecondStage(vit_model, feature_extractor).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Assuming binary classification for each of the 3 classes
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load model from checkpoint if available
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")

    # Load datasets
    train_dataset = SecondStageDataset(train_image_files, label_file, train_first_stage_predictions_csv, train=True)
    val_dataset = SecondStageDataset([val_image_file], label_file, val_first_stage_predictions_csv, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        running_loss = 0.0

        for i, (images, labels, first_stage_preds) in enumerate(train_loader):
            images, labels, first_stage_preds = images.to(device), labels.to(device), first_stage_preds.to(device)

            optimizer.zero_grad()
            outputs = model(images, first_stage_preds)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Loss: {running_loss/len(train_loader)}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'disease_detector_second_stage_checkpoint.pth')

    print("Second stage training complete.")

    # Save the trained model
    torch.save(model.state_dict(), 'disease_detector_second_stage_model.pth')
    print("Second stage model saved to 'disease_detector_second_stage_model.pth'.")

train_image_files = [f'/group/pmc023/rnandiya/region_train_2_npz/chunk{i}.npz' for i in range(1, 13)]
label_file = 'final_data.json'  
val_image_file = "/group/pmc023/rnandiya/validation2.npz"
train_first_stage_predictions_csv = 'predictions_training.csv'
val_first_stage_predictions_csv = 'predictions_validation.csv'

train_second_stage_model(train_image_files, label_file, val_image_file, 
                         train_first_stage_predictions_csv, val_first_stage_predictions_csv, 
                         num_epochs=40, batch_size=1000, learning_rate=0.07, 
                         checkpoint_path='disease_detector_second_stage_checkpoint.pth')
