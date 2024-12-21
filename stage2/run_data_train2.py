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

# Function to calculate class weights
def calculate_class_weights(dataset):
    labels = dataset.labels  # Extract labels from dataset
    class_counts = labels.sum(axis=0)  # Sum each class across the dataset
    total_samples = labels.shape[0]  # Total number of samples
    class_weights = total_samples / (class_counts + 1e-5)  # Calculate weights
    return torch.tensor(class_weights, dtype=torch.float32)

def train_second_stage_model(train_image_files, train_first_stage_pred_files, train_second_stage_pred_files, label_train_file, 
                             val_image_files, val_first_stage_pred_files, val_second_stage_pred_files, label_val_file,
                             num_epochs=1, batch_size=16, learning_rate=1e-4, 
                             checkpoint_path=None, device=None):
    # Load the pre-trained ViT model and feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Initialize the model, loss function, and optimizer
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiseaseDetectorSecondStage(vit_model, feature_extractor).to(device)

    # Load datasets
    train_dataset = SecondStageDataset(train_image_files, train_first_stage_pred_files, train_second_stage_pred_files, label_train_file)
    val_dataset = SecondStageDataset(val_image_files, val_first_stage_pred_files, val_second_stage_pred_files, label_val_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset).to(device)
    
    # Use weighted loss to handle class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load model from checkpoint if available
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")

    # Lists to store loss and accuracy
    train_losses, val_losses = [], []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training
        model.train()
        for i, (images, first_stage_preds, second_stage_preds, labels) in enumerate(train_loader):
            images, first_stage_preds, second_stage_preds, labels = images.to(device), first_stage_preds.to(device), second_stage_preds.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, first_stage_preds, second_stage_preds)
            loss = criterion(outputs, labels)  # Use the weighted loss function
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            correct_train += accuracy.item() * labels.size(0)
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, first_stage_preds, second_stage_preds, labels in val_loader:
                images, first_stage_preds, second_stage_preds, labels = images.to(device), first_stage_preds.to(device), second_stage_preds.to(device), labels.to(device)

                outputs = model(images, first_stage_preds, second_stage_preds)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()

                accuracy = calculate_accuracy(outputs, labels)
                correct_val += accuracy.item() * labels.size(0)
                total_val += labels.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

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

    # Plotting loss and accuracy
    epochs_range = range(start_epoch, start_epoch + num_epochs)
    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    # Save plot
    plt.tight_layout()
    plt.savefig('training_validation_loss_accuracy.jpg')
    plt.show()

# Example usage:
train_image_files = '/group/pmc023/rnandiya/train_s2_image.npz'
train_first_stage_pred_files = '/group/pmc023/rnandiya/train_s2_pred_curr.npz'
train_second_stage_pred_files = '/group/pmc023/rnandiya/train_match_pred.npz'
label_train_file = '/group/pmc023/rnandiya/train_s2_label.npz'

val_image_files = '/group/pmc023/rnandiya/val_s2_image.npz'
val_first_stage_pred_files = '/group/pmc023/rnandiya/val_s2_pred_curr.npz'
val_second_stage_pred_files = '/group/pmc023/rnandiya/val_match_pred.npz'
label_val_file = '/group/pmc023/rnandiya/val_s2_label.npz'

train_second_stage_model(train_image_files, train_first_stage_pred_files, train_second_stage_pred_files, label_train_file,
                         val_image_files, val_first_stage_pred_files, val_second_stage_pred_files, label_val_file,
                         num_epochs=70, batch_size=500, learning_rate=1e-4, 
                         checkpoint_path='disease_detector_second_stage_checkpoint.pth')
