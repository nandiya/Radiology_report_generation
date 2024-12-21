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

def concatenate_images(images, grid_size=(1, 2)):
    """Concatenate images into a single image in a 1x2 grid."""
    rows = []
    for i in range(0, len(images), grid_size[1]):
        row_images = images[i:i + grid_size[1]]
        row = np.concatenate([np.array(img) for img in row_images], axis=1)
        rows.append(row)
    concatenated_image = np.concatenate(rows, axis=0)
    return concatenated_image

def process_image(image_path, prev_image_path):
    """Process the image and return the concatenated result as a Tensor."""
    image = load_image(image_path)
    prev_image = load_image(prev_image_path)
    
    resized_prev_image = resize_image(prev_image, size=(224, 224))
    resized_whole_image = resize_image(image, size=(224, 224))
    
    # Include the resized whole image and the previous image
    cropped_images = [resized_whole_image, resized_prev_image]
    concatenated_image = concatenate_images(cropped_images, grid_size=(1, 2))
    torch_image = torch.tensor(concatenated_image, dtype=torch.float32).permute(2, 0, 1)
    return torch_image

def process_json_files(final_data_json_file, match_json_file, root_path):
    combined_images = []
    
    # Load the JSON data
    with open(final_data_json_file, 'r') as final_file:
        final_data = json.load(final_file)

    with open(match_json_file, 'r') as match_file:
        match_data = json.load(match_file)
    
    # Iterate over the matched image IDs and process the images
    for img_id in match_data:
        # Get the current image path from final_data.json
        try:

            image_path = root_path+final_data["test"][img_id]["disease"]["image_path"][0]
        except KeyError:
            print(img_id)
            continue  # Skip if the image ID is not found in final_data.json

        prev_image_path = match_data[img_id]["source_prev"]
        # Ensure both image files exist
        if os.path.exists(image_path) and os.path.exists(prev_image_path):
   
            # Process the images and concatenate them
            torch_image = process_image(image_path, prev_image_path)
            combined_images.append(torch_image)
        break
    # Stack all images into a single numpy array and save as .npz file
    if combined_images:
        combined_images_tensor = torch.stack(combined_images)
        np.savez('s2_test.npz', data=combined_images_tensor.numpy())

# Update paths to the JSON files and root image directory
final_data_json_file = "final_data.json"
match_json_file = "image_match_source_test.json"
root_path = "/group/pmc023/rnandiya/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"  # Update if needed

# Process the JSON files
process_json_files(final_data_json_file, match_json_file, root_path)
