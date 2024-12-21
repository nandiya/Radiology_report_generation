import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ViTFeatureExtractor, ViTModel
import torch.nn as nn
import torch.nn.functional as F

class MultiFileDataset(Dataset):
    def __init__(self, image_npz_files):
        self.image_data = []

        # Load image data from npz files
        for npz_file in image_npz_files:
            data = np.load(npz_file, allow_pickle=True)
            for key in data.files:
                self.image_data.extend(data[key])
        
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        return image

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

def evaluate_model(test_image_file, model_path, batch_size=16, device=None):
    # Load the pre-trained ViT model and feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Initialize the model and load the trained weights
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = diseaseDetector(vit_model, feature_extractor).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the test dataset (no labels)
    test_dataset = MultiFileDataset(test_image_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []

    # Make predictions on the test dataset
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = (outputs > 0).float()
            all_preds.extend(preds.cpu().numpy())

    all_preds = np.array(all_preds)

    # Save predictions to a npz file
    npz_file = "val_match_pred.npz"
    np.savez(npz_file, predictions=all_preds)

    print(f"Predictions saved to {npz_file}")

# Example usage:
test_image_file = ["/group/pmc023/rnandiya/val_s2_s1.npz"]
model_path = '/group/pmc023/rnandiya/model/disease_detector_model12.pth'
evaluate_model(test_image_file, model_path, batch_size=1000)
