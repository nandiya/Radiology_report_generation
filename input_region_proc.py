import json
import os
import numpy as np
from PIL import Image
import torch

def load_image(image_path):
    """Load an image from file."""
    return Image.open(image_path).convert("RGB")

def resize_image(image, size=(448, 448)):
    """Resize the image to the given size."""
    return image.resize(size, Image.Resampling.LANCZOS)

def crop_and_resize(image, bboxes, crop_size=(224, 224)):
    """Crop and resize each bounding box region to the given size."""
    cropped_images = []
    for (x, y, w, h) in bboxes:
        cropped_region = image.crop((x, y, x + w, y + h))
        resized_region = cropped_region.resize(crop_size, Image.Resampling.LANCZOS)
        cropped_images.append(resized_region)
    return cropped_images

def concatenate_images(images, grid_size=(1, 4)):
    """Concatenate images into a single image in a 1x4 grid."""
    rows = []
    for i in range(0, len(images), grid_size[1]):
        row_images = images[i:i + grid_size[1]]
        row = np.concatenate([np.array(img) for img in row_images], axis=1)
        rows.append(row)
    concatenated_image = np.concatenate(rows, axis=0)
    return concatenated_image

def process_image(image_path, bboxes):
    """Process the image and return the concatenated result as a TensorFlow tensor."""
    image = load_image(image_path)
    resized_image = resize_image(image, size=(448, 448))
    resized_whole_image = resize_image(image, size=(224, 224))
    # Include the resized whole image
    cropped_images = [resized_whole_image] + crop_and_resize(resized_image, bboxes, crop_size=(224, 224))
    concatenated_image = concatenate_images(cropped_images, grid_size=(1, 5))
    torch_image = torch.tensor(concatenated_image, dtype=torch.float32).permute(2, 0, 1)
    return torch_image

def process_json_files(bbox_json_file, path_json_file, root_path):
    combined_images = []
    with open(bbox_json_file, 'r') as bbox_file:
        bbox_data = json.load(bbox_file)

    with open(path_json_file, 'r') as path_file:
        path_data = json.load(path_file)
    a = 1
    for img_id, bboxes_dict in bbox_data.items():
        if img_id in path_data["test"]:
            image_path = os.path.join(root_path, path_data["test"][img_id]["disease"]["image_path"][0])
    
            if os.path.exists(image_path):  # Ensure the image file exists
                bboxes = bboxes_dict
                torch_image = process_image(image_path, bboxes["bboxes"])
                combined_images.append(torch_image)
       
        
    
    # Stack all images into a single numpy array and save as .npy file
    combined_images_tensor = torch.stack(combined_images)
    np.savez('test_data2.npz', data=combined_images_tensor.numpy())

# Example usage:
bbox_json_file = "test_regions_bb_4.json"  # Update this to your JSON file containing bounding box data
path_json_file = "final_data.json"  # Update this to your JSON file containing image paths
root_path = "/group/pmc023/rnandiya/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"  # Update this to the root path of your images

process_json_files(bbox_json_file, path_json_file, root_path)
