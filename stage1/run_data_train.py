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

class MultiFileDataset(Dataset):
    def __init__(self, image_npz_files, label_file, train=True):
        self.image_data = []
        self.labels = []

        # Load image data from npz files
        for npz_file in image_npz_files:
            data = np.load(npz_file, allow_pickle=True)
            for key in data.files:
                print(f"  {key}: {data[key].shape}")
                self.image_data.extend(data[key])
        
        # Print length of image data for debugging
        print(f"Total images loaded: {len(self.image_data)}")

        # Load labels from the single JSON file
        with open(label_file, 'r') as f:
            labels = json.load(f)
            if train:
                for img_id in labels["train"]:
                    self.labels.append(labels["train"][img_id]["disease"]["disease"])
            else:
                for img_id in labels["val"]:
                    self.labels.append(labels["val"][img_id]["disease"]["disease"])
        
        # Print length of labels for debugging
        print(f"Total labels loaded: {len(self.labels)}")

        # Check if the number of images matches the number of labels
        if len(self.image_data) != len(self.labels):
            raise ValueError("The number of images and labels do not match.")
        
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.float32)


import torch
import torch.nn as nn

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
        #self.sigmoid = nn.Sigmoid()

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
        result = F.tanh(combined)

        return result

import matplotlib.pyplot as plt

def train_model(train_image_files, label_file, val_image_file, num_epochs=1, batch_size=16, learning_rate=1e-4, 
                checkpoint_path=None, device=None):
    # Load the pre-trained ViT model and feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Initialize the model, loss function, and optimizer
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        print("Using CPU")
    model = diseaseDetector(vit_model, feature_extractor).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for multi-label classification
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
    train_dataset = MultiFileDataset(train_image_files, label_file, train=True)
    val_dataset = MultiFileDataset([val_image_file], label_file, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"Starting epoch {epoch+1}")

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.numel()

            # print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item()}")

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)

        # Validation step
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (outputs > 0.2).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.numel()

                # print(f"Validation Batch {i+1}/{len(val_loader)}, Loss: {loss.item()}")

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct_val / total_val)
        model.train()

        print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")
        print(f"Training Accuracy: {correct_train/total_train}, Validation Accuracy: {correct_val/total_val}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'disease_detector_checkpoint.pth')

    print("Training complete.")

    # Save the trained model
    torch.save(model.state_dict(), 'disease_detector_model.pth')
    print("Model saved to 'disease_detector_model.pth'.")

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(start_epoch, start_epoch + num_epochs), train_losses, label='Training Loss')
    plt.plot(range(start_epoch, start_epoch + num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')

    # Plot training and validation accuracy
    plt.figure()
    plt.plot(range(start_epoch, start_epoch + num_epochs), train_accuracies, label='Training Accuracy')
    plt.plot(range(start_epoch, start_epoch + num_epochs), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')



image_files = [f'/group/pmc023/rnandiya/region_train_2_npz/chunk{i}.npz' for i in range(1, 13)]  # List of all data files
label_file = 'final_data.json'  
val_image_file = "/group/pmc023/rnandiya/validation2.npz"
train_model(image_files, label_file, val_image_file ,num_epochs=30, batch_size=1000, learning_rate=0.05, 
            checkpoint_path='/group/pmc023/rnandiya/disease_detector_checkpoint12.pth')