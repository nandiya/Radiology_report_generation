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

def filter_regions(image, regions, min_area=40000, max_area=55000, max_background_ratio=0.7):
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

def mirror_bounding_box(box, image_width):
    x, y, w, h = box
    mirrored_x = image_width - x - w
    return (mirrored_x, y, w, h)

def shift_bounding_box(box, shift_x, shift_y, image_width, image_height):
    x, y, w, h = box
    if x + shift_x < 0 or x + w + shift_x > image_width:
        shift_x = -shift_x
    if y + shift_y < 0 or y + h + shift_y > image_height:
        shift_y = -shift_y
    new_x = min(image_width - w, max(0, x + shift_x))
    new_y = min(image_height - h, max(0, y + shift_y))
    return (new_x, new_y, w, h)

# Initialize variables
root_path = "/group/pmc023/rnandiya/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
imageId_disease = json.load(open("/group/pmc023/rnandiya/data_result/train_chunk_9.json", "r", encoding="utf-8"))
list_regions = {}
# print("")
# img = load_and_resize_image(root_path, imageId_disease["7f23b996-22544258-fcf2fbc3-f8dbf8e7-b6c0e4c5"]["disease"]["image_path"][0])
# image_height, image_width, _ = img.shape

# regions = selective_search(img)
# filtered_regions = filter_regions(img, regions)
# # Ensure the filtered regions are converted to Python int
# filtered_regions = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in filtered_regions]
# if len(filtered_regions) == 4:
#     list_regions[img_id] = filtered_regions
# else:
#     list_regions[img_id] = filtered_regions[:4]
# while len(filtered_regions) < 4:
#     if len(filtered_regions) == 1:
#         mirrored_box = mirror_bounding_box(filtered_regions[0], image_width)
#         filtered_regions.append(mirrored_box)
#     elif len(filtered_regions) == 2:
#         shifted_box_right = shift_bounding_box(filtered_regions[0], shift_x=100, shift_y=0, image_width=image_width, image_height=image_height)
#         filtered_regions.append(shifted_box_right)
#     elif len(filtered_regions) == 3:
#         shifted_box_down = shift_bounding_box(filtered_regions[0], shift_x=0, shift_y=100, image_width=image_width, image_height=image_height)
#         filtered_regions.append(shifted_box_down)

# if len(filtered_regions) > 4:
#     filtered_regions = filtered_regions[:4]

# print(filtered_regions)
# for (x, y, w, h) in filtered_regions:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # Save the image with bounding boxes
# output_image_path = "output_image_with_boxes1.jpg"
# cv2.imwrite(output_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Save regions to json file
# file_name = "regions_" + img_id + ".json"
# with open(file_name, "w") as outfile:
#     json.dump(list_regions, outfile)
for img_id in imageId_disease:
    try:
        img = load_and_resize_image(root_path, imageId_disease[img_id]["disease"]["image_path"][0])
        image_height, image_width, _ = img.shape
        regions = selective_search(img)
        filtered_regions = filter_regions(img, regions)
        # Ensure the filtered regions are converted to Python int
        filtered_regions = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in filtered_regions]
        # while len(filtered_regions) < 4:
        #     if len(filtered_regions) == 1:
        #         mirrored_box = mirror_bounding_box(filpwtered_regions[0], image_width)
        #         filtered_regions.append(mirrored_box)
        #     elif len(filtered_regions) == 2:
        #         shifted_box_right = shift_bounding_box(filtered_regions[0], shift_x=100, shift_y=0, image_width=image_width, image_height=image_height)
        #         filtered_regions.append(shifted_box_right)
        #     elif len(filtered_regions) == 3:
        #         shifted_box_down = shift_bounding_box(filtered_regions[0], shift_x=0, shift_y=100, image_width=image_width, image_height=image_height)
        #         filtered_regions.append(shifted_box_down)

        if len(filtered_regions) > 4:
            filtered_regions = filtered_regions[:4]
        list_regions[img_id] = filtered_regions
    except:
        print(root_path)

file_name = "origin_regions_chunk9.json"
with open(file_name, "w") as outfile:
    json.dump(list_regions, outfile)
