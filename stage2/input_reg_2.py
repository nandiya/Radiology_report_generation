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

def process_image(image_path, curr_img):
   
    """Process the image and return the concatenated result as a TensorFlow tensor."""
    image = load_image(image_path)
    image_curr = load_image(curr_img)

    resized_whole_image = resize_image(image, size=(224, 224))
    resized_whole_image2 = resize_image(image_curr, size=(224, 224))
    # Include the resized whole image
    cropped_images = [resized_whole_image] + [resized_whole_image2]
    concatenated_image = concatenate_images(cropped_images, grid_size=(1, 2))
    torch_image = torch.tensor(concatenated_image, dtype=torch.float32).permute(2, 0, 1)
    return torch_image

def process_json_files(bbox_json_file, path_json_file,path_curr_image_file, root_path):
    combined_images = []
    with open(bbox_json_file, 'r') as bbox_file:
        bbox_data = json.load(bbox_file)

    with open(path_json_file, 'r') as path_file:
        path_data = json.load(path_file)

    with open(path_curr_image_file, 'r') as path2_file:
        curr_data = json.load(path2_file)

    for img_id, bboxes_dict in bbox_data.items():
        
        # Iterate through the path_data to find the "prev" matching img_id
        for key, value in path_data.items():
            
            if value['prev'] == img_id:
                
                image_path = os.path.join(root_path, value["source_prev"])
                curr_image_path = os.path.join(root_path,curr_data["val"][key]["disease"]["image_path"][0])
                if os.path.exists(image_path):  
                    bboxes = bboxes_dict
                    torch_image = process_image(image_path,curr_image_path)
                    combined_images.append(torch_image)
        

    if combined_images:
        combined_images_tensor = torch.stack(combined_images)
        np.savez('val_s2_image.npz', data=combined_images_tensor.numpy())
    else:
        print("No images processed.")

# Example usage
bbox_json_file = "val_s2_bb4s.json"
path_json_file = "image_match_source_val.json"
path_curr_image_file = "final_data.json"
root_path = "/group/pmc023/rnandiya/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
process_json_files(bbox_json_file, path_json_file,path_curr_image_file, root_path)



