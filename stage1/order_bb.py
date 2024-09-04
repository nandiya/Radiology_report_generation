import json

# Load the JSON data from a file (assuming the filename is 'bboxes.json')
with open('train_bb_4.json', 'r') as file:
    data = json.load(file)

# Function to get the top-left corner of a bounding box
def top_left_corner(bbox):
    x, y, width, height = bbox
    return (x, y)

# Function to get the center of a bounding box
def center(bbox):
    x, y, width, height = bbox
    return (x + width / 2, y + height / 2)

# Function to sort bounding boxes based on their position
def sort_bboxes(bboxes):
    bboxes.sort(key=lambda bbox: (center(bbox)[1], center(bbox)[0]))  # Sort by y first, then by x
    top_half = sorted(bboxes[:len(bboxes)//2], key=lambda bbox: top_left_corner(bbox)[0])
    bottom_half = sorted(bboxes[len(bboxes)//2:], key=lambda bbox: top_left_corner(bbox)[0])
    return top_half + bottom_half

# Process each image in the JSON data
for image_id, image_data in data.items():
    bboxes = image_data['bboxes']
    sorted_bboxes = sort_bboxes(bboxes)
    data[image_id]['bboxes'] = sorted_bboxes

# Save the sorted bounding boxes back to a new JSON file
with open('train_bb_4.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Bounding boxes sorted and saved to 'sorted_bboxes.json'")
