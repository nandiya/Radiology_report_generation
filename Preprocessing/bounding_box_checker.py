import os
import json

def mirror_bounding_box(bbox, image_width):
    """Mirror the bounding box horizontally."""
    x, y, w, h = bbox
    mirrored_x = image_width - x - w
    return (mirrored_x, y, w, h)

def shift_bounding_box(bbox, image_width, shift_amount):
    """Shift the bounding box horizontally, reversing direction if near the edge."""
    x, y, w, h = bbox
    if x + w + shift_amount < image_width:
        shifted_x = x + shift_amount
    else:
        shifted_x = max(0, x - shift_amount)
    return (shifted_x, y, w, h)

def check_overlap(bbox1, bbox2):
    """Check if two bounding boxes overlap."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    overlap = not (x1 >= x2 + w2 or x1 + w1 <= x2 or y1 >= y2 + h2 or y1 + h1 <= y2)
    return overlap

def ensure_four_bounding_boxes(bboxes, image_width, image_height):
    """Ensure there are at least four bounding boxes, adding mirrored and shifted versions if necessary."""
    new_bboxes = bboxes.copy()
    shift_amount = 20

    # Add mirrored bounding boxes if fewer than 4
    for bbox in bboxes:
        if len(new_bboxes) >= 4:
            break
        mirrored_bbox = shift_bounding_box(bbox, image_width, shift_amount)
        new_bboxes.append(mirrored_bbox)

    # Add more shifted bounding boxes if still fewer than 4
    return new_bboxes

def process_json_file(file_path, combined_data, image_width, image_height, previous_bboxes=None):
    with open(file_path, 'r') as file:
        data = json.load(file)

    for img_id in data:
        bboxes = data[img_id].get("bboxes", [])

        if len(bboxes) == 0 and previous_bboxes:
            # Copy previous bboxes if current is empty
            bboxes = previous_bboxes
        elif len(bboxes) < 4:
            # Ensure there are at least 4 bboxes
            bboxes = ensure_four_bounding_boxes(bboxes, image_width, image_height)
        
        combined_data[img_id] = {
            "bboxes": bboxes
        }

        # Store the current bboxes as previous for the next iteration
        previous_bboxes = bboxes

def main():
    input_folder = "/group/pmc023/rnandiya/origin_train/"  
    output_file = "val_s2_bb4s.json"  # Update this to your desired output file
    
    image_width = 448  
    image_height = 448  
    
    combined_data = {}
    previous_bboxes = None
    
    process_json_file("val_s2_bb4s.json", combined_data, image_width, image_height, previous_bboxes)
    
    with open(output_file, 'w') as file:
        json.dump(combined_data, file, indent=4)

if __name__ == "__main__":
    main()
