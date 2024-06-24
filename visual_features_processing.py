
from transformers import AutoImageProcessor, ViTModel
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
from collections import Counter
import json
import torch
import pandas as pd
import os

def load_images(root_path, image_paths):
    img_path = os.path.join(root_path, image_paths)
    image = Image.open(img_path).convert("RGB")
    #image = image.resize(new_size)
    image = np.array(image)
    return image

def get_rois_and_locs( rects):
        rois = []
        locs = []
        for (x, y, w, h) in rects:
            if w / float(W) < 0.1 or h / float(H) < 0.1:
                continue
            roi = orig[y:y + h, x:x + w]
            roi = cv2.resize(roi, self.kwargs['INPUT_SIZE'])
            rois.append(roi)
            locs.append((x, y, w, h))
        return rois, locs
    
def visualize_rois(rois):
    fig, axes = plt.subplots(1, len(rois), figsize=(20, 6))
    for ax, roi in zip(axes, rois):
        ax.imshow(roi, cmap='gray')


def selective_search(image, method="fast"):
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        print("12")
        ss.setBaseImage(image)
        print("22")
        ss.switchToSelectiveSearchFast()
    
        return ss.process()

def calculate_background_ratio(region):
    """
    Calculate the background ratio of a region.
    The background is assumed to be the most common color in the region.
    """
    region = region.reshape(-1, region.shape[-1])
    color_counts = Counter(map(tuple, region))
    background_color, background_count = color_counts.most_common(1)[0]
    background_ratio = background_count / float(region.shape[0])
    return background_ratio

def iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def filter_regions(image, regions, min_area=800000, max_area=2500000, max_background_ratio=0.7):
    filtered_regions = []
    seen_regions = set()
    
    for (x, y, w, h) in regions:
        area = w * h
        check_ratio = 0
        if w>h :
            check_ratio = w/h
        else:
            check_ratio = h/w
        if area < min_area or area > max_area:
            continue
        if check_ratio > 8:
            continue
        region = image[y:y+h, x:x+w]
        background_ratio = calculate_background_ratio(region)
        if background_ratio > max_background_ratio:
            continue
        is_similar = False
        for (fx, fy, fw, fh) in filtered_regions:
            if abs(w - fw) <= 3 and abs(h - fh) <= 3:
                is_similar = True
                break
        if is_similar:
            continue

        # Check if the region overlaps significantly with any already accepted region
        overlaps = False
        for (fx, fy, fw, fh) in filtered_regions:
            if iou((x, y, w, h), (fx, fy, fw, fh)) > 0.3:
                overlaps = True
                break
        if overlaps:
            continue

        region_key = (x, y, w, h)
        if region_key in seen_regions:
            continue
        seen_regions.add(region_key)
        filtered_regions.append((x, y, w, h))
    
    return filtered_regions
    

def visualize_and_save(image, regions, output_path):
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# def region_visual_extractor(image, regions, size=(224, 224), image_processor=None, model=None):
#     concatenated_features = None
    
#     for (x, y, w, h) in regions:
#         region = image[y:y+h, x:x+w]
#         region_resized = cv2.resize(region, size)
        
#         inputs = image_processor(images=region_resized, return_tensors="pt")
#         with torch.no_grad():
#             outputs = model(**inputs)
#         last_hidden_states = outputs.last_hidden_state

#         if concatenated_features is None:
#             concatenated_features = last_hidden_states
#         else:
#             concatenated_features = torch.cat((concatenated_features, last_hidden_states))
    
#     return concatenated_features




root_path = "/group/pmc023/rnandiya/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"

imageId_disease = json.load(open("imageId_disease.json", "r", encoding="utf-8"))

# feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
# model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
j  = 1
list_regions = {}
for data in imageId_disease:
    tmp_dict = {}
    for img_id in imageId_disease[data]:
        print(j)
        print("--")
        img = load_images(root_path, imageId_disease[data][img_id]["image_path"][0])
        print("00")
        regions = selective_search(img)
        print("aaa")
        filtered_regions = filter_regions(img, regions)
        print(filtered_regions)
        #region_feature_results = region_visual_extractor(img,filtered_regions, (224, 224), feature_extractor, model)
        #visualize_an d_save(img, filtered_regions, "output_image.jpg")
        if len(filtered_regions) == 4:
            tmp_dict[img_id] = filtered_regions
        else:
            print("1")
            tmp_dict[img_id] = filtered_regions[:4]
            print("ssss")
        #(region_feature_results.shape)
        j = j+1
        break
    list_regions[data] = tmp_dict
    break   

with open("list_regions.json", "w") as outfile: 
    json.dump(list_regions, outfile)




