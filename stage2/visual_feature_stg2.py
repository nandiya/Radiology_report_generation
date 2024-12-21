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

def concatenate_images(images, grid_size=(1, 6)):
    """Concatenate images into a single image in a 1x6 grid."""
    rows = []
    for i in range(0, len(images), grid_size[1]):
        row_images = images[i:i + grid_size[1]]
        row = np.concatenate([np.array(img) for img in row_images], axis=1)
        rows.append(row)
    concatenated_image = np.concatenate(rows, axis=0)
    return concatenated_image

def find_image_file(root_path, image_id):
    """Search for an image file by image_id in all folders within root_path."""
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename == f"{image_id}.jpg":
                return os.path.join(dirpath, filename)
    return None

def process_image(image_path, prev_image_path, bboxes):
    """Process the image and return the concatenated result as a Tensor."""
    image = load_image(image_path)
    prev_image = load_image(prev_image_path)
    
    resized_image = resize_image(image, size=(448, 448))
    resized_prev_image = resize_image(prev_image, size=(224, 224))
    resized_whole_image = resize_image(image, size=(224, 224))
    
    # Include the resized whole image and the previous image
    cropped_images = [resized_whole_image] + crop_and_resize(resized_image, bboxes, crop_size=(224, 224)) + [resized_prev_image]
    concatenated_image = concatenate_images(cropped_images, grid_size=(1, 6))
    torch_image = torch.tensor(concatenated_image, dtype=torch.float32).permute(2, 0, 1)
    return torch_image

def process_json_files(bbox_json_file, all_json_file, chunk_json_file, match_json_file, root_path):
    combined_images = []
    with open(bbox_json_file, 'r') as bbox_file:
        bbox_data = json.load(bbox_file)
    
    with open(chunk_json_file, 'r') as chunk_file:
        path_data = json.load(chunk_file)

    with open(all_json_file, 'r') as all_js_file:
        all_file = json.load(all_js_file)

    with open(match_json_file, 'r') as match_file:
        match_data = json.load(match_file)
    
    for img_id, bboxes_dict in bbox_data.items():
        if img_id in match_data["train"]:
            try:
                image_path = os.path.join(root_path, all_file["train"][img_id]["disease"]["image_path"][0])
            except:
                image_path = find_image_file(root_path, img_id)
            # Get the previous image ID from the match JSON file
            prev_img_id = match_data["train"].get(img_id, {}).get("prev", None)
            if prev_img_id:
                print("ssss")
                try:
                    prev_image_path = os.path.join(root_path, all_file["train"][prev_img_id]["disease"]["image_path"][0])
                except:
                    prev_image_path = None
                # If the previous image path is not found, search for the file in the root_path
                if prev_image_path is None:
                    prev_image_path = find_image_file(root_path, prev_img_id)
                
                if os.path.exists(image_path) and prev_image_path and os.path.exists(prev_image_path):  # Ensure both image files exist
                    bboxes = bboxes_dict
                    torch_image = process_image(image_path, prev_image_path, bboxes["bboxes"])
                    combined_images.append(torch_image)

        
    # Stack all images into a single numpy array and save as .npy file
    combined_images_tensor = torch.stack(combined_images)
    np.savez('s2_chunk5.npz', data=combined_images_tensor.numpy())



bbox_json_file = "train_bb_4s.json"
all_json_file = "/group/pmc023/rnandiya/dataset/mimic_annotation_all.json"
chunk_json_file = "/group/pmc023/rnandiya/data_result/train_chunk_5.json"  # Update this to the specific JSON chunk you want to process
match_json_file = "image_match.json"  # Update this to your JSON file
root_path = "/group/pmc023/rnandiya/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"  # Update this to the root path of your images

process_json_files(bbox_json_file, all_json_file, chunk_json_file, match_json_file, root_path)