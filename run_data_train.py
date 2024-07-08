import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import torchvision.models as models
from torchvision import transforms
from transformers import PreTrainedModel, ViTModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from transformers import ViTConfig


class diseaseDetector(nn.Module):
    def __init__(self, model, image_processor, ):
        super(diseaseDetector, self).__init__()
        self.model = model
        self.image_processor = image_processor
        self.resize_transform = transforms.Resize((224, 224))
        self.linear = nn.Linear(786,56)
        self.linear2 = nn.Linear(56, 14)
        self.sigmoid = nn.Sigmoid(dim=1)

    def visual_extractor(self, region):
        inputs = self.image_processor(images=region, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states
    
    def region_cropper(self, image, bb):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        cropped_regions = []
        for box in bb:
            x, y, width, height = box
            region = image.crop((x, y, x + width, y + height))
            region = self.resize_transform(region)  # Resize the cropped region
            cropped_regions.append(region)
        return cropped_regions

    def forward(self, input,bb,):
        # cropped_regions = self.region_cropper(input, bb)

        # for region in cropped_regions:
        #     region_features = self.visual_extractor(region)
            
        region1 = self.visual_extractor(input[:,:,224])
        region2 = self.visual_extractor(input[:,:,224:448])
        region3 = self.visual_extractor(input[:,:, 448:672])
        region4 = self.visual_extractor(input[:,:,672:896])

        disease1 = self.linear(region1)
        disease2 = self.linear(region2)
        disease3 = self.linear(region3)
        disease4 = self.linear(region4)

        combined = torch.cat((disease1, disease2, disease3, disease4), dim=1)
        result = self.linear2(combined)
        result = self.sigmoid(result)

        return result


