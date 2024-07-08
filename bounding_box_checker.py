import os
import json

def mirror_bounding_box(bbox, image_width):
    """Mirror the bounding box horizontally."""
    x, y, w, h = bbox
    mirrored_x = image_width - x - w
    return (mirrored_x, y, w, h)

def shift_bounding_box(bbox, image_width, shift_amount, shifted_bboxes):
    """Shift the bounding box horizontally, reversing direction if near the edge."""
    x, y, w, h = bbox
    if x + w + shift_amount < image_width and all(new_x + new_w <= image_width for new_x, _, new_w, _ in shifted_bboxes):
        shifted_x = x + shift_amount
    else:
        shifted_x = max(0, x - shift_amount)
    return (shifted_x, y, w, h)

def check_overlap(bbox1, bbox2):
    """Check if two bounding boxes overlap."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 >= x2 + w2 or x1 + w1 <= x2 or y1 >= y2 + h2 or y1 + h1 <= y2)

def ensure_four_bounding_boxes(bboxes, image_width, image_height):
    """Ensure there are at least four bounding boxes, adding mirrored and shifted versions if necessary."""
    new_bboxes = bboxes.copy()
    
    # Add mirrored bounding boxes if fewer than 4
    for bbox in bboxes:
        if len(new_bboxes) >= 4:
            break
        mirrored_bbox = mirror_bounding_box(bbox, image_width)
        if not any(check_overlap(mirrored_bbox, existing_bbox) for existing_bbox in new_bboxes):
            new_bboxes.append(mirrored_bbox)
    
    # Add shifted bounding boxes if still fewer than 4
    shift_amount = 20
    while len(new_bboxes) < 4:
        shifted_bboxes = new_bboxes.copy()
        for bbox in bboxes:
            if len(new_bboxes) >= 4:
                break
            shifted_bbox = shift_bounding_box(bbox, image_width, shift_amount, shifted_bboxes)
            if (shifted_bbox[0] + shifted_bbox[2] <= image_width and
                shifted_bbox[1] + shifted_bbox[3] <= image_height and
                not any(check_overlap(shifted_bbox, existing_bbox) for existing_bbox in new_bboxes)):
                new_bboxes.append(shifted_bbox)
        shift_amount += 20
    
    return new_bboxes[:4]

def process_json_file(file_path, combined_data, image_width, image_height):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    for img_id in data:
        bboxes = data[img_id]
        
        if len(bboxes) < 4:
            bboxes = ensure_four_bounding_boxes(bboxes, image_width, image_height)
        
        combined_data[img_id] = {
            "bboxes": bboxes
        }
        break

def main():
    input_folder = "/group/pmc023/rnandiya/origin_train/"  # Update this to your JSON files folder
    output_file = "train_bb.json"  # Update this to your desired output file
    
    image_width = 448  # Set the image width to 448
    image_height = 448  # Set the image height to 448
    
    combined_data = {}
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            process_json_file(file_path, combined_data, image_width, image_height)
            break
    
    with open(output_file, 'w') as file:
        json.dump(combined_data, file, indent=4)

if __name__ == "__main__":
    main()
