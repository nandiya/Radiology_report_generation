import json
import os
import numpy as np
from PIL import Image
import cv2
import argparse
from collections import Counter


def load_and_resize_image(root_path, image_path, size=(448, 448)):
    img_path = os.path.join(root_path, image_path)
    image = Image.open(img_path).convert("RGB")
    image = image.resize(size, Image.LANCZOS)
    image = np.array(image)
    return image


def calculate_background_ratio(region):
    region = region.reshape(-1, region.shape[-1])
    color_counts = Counter(map(tuple, region))
    background_color, background_count = color_counts.most_common(1)[0]
    background_ratio = background_count / float(region.shape[0])
    return background_ratio

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def filter_regions(image, regions, min_area=38000, max_area=55000, max_background_ratio=0.7):
    filtered_regions = []
    seen_regions = set()
    for (x, y, w, h) in regions:
        area = w * h
        check_ratio = 0
        if w > h:
            check_ratio = w / h
        else:
            check_ratio = h / w
        if area < min_area or area > max_area:
            continue
        if check_ratio > 4:
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

def selective_search(image, method="fast"):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if method == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    return ss.process()

# Initialize variables
root_path = "/group/pmc023/rnandiya/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
## since the execution is done in 12 cpu in kaya, change the file name into each file in split training manually
## for ex : imageId_disease = json.load(open("chunk1.json", "r", encoding="utf-8"))
imageId_disease = json.load(open("image_match_source_train.json", "r", encoding="utf-8"))
list_regions = {}


a = 1

## for img_id in imageId_disease : // for training 
for img_id in imageId_disease:

    img = load_and_resize_image(root_path, imageId_disease[img_id]["source_prev"])
    image_height, image_width, _ = img.shape
    regions = selective_search(img)
    filtered_regions = filter_regions(img, regions)

    filtered_regions = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in filtered_regions]

    if len(filtered_regions) > 4:
        filtered_regions = filtered_regions[:4]
    list_regions[img_id] = filtered_regions
    break


file_name = "trai_s2_bb.json"
with open(file_name, "w") as outfile:
   json.dump(list_regions, outfile)
