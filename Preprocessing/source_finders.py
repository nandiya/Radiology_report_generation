import json

# Function to read JSON file
def read_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# Function to read txt file with image paths
def read_txt(txt_file):
    with open(txt_file, 'r') as f:
        return f.readlines()

# Function to find the source of prev image id
def find_prev_image_source(json_data, txt_data):
    # Create a dictionary to map image IDs to their paths
    image_id_to_path = {}
    for path in txt_data:
        path = path.strip()
        image_id = path.split('/')[-1].replace('.jpg', '')
        image_id_to_path[image_id] = path

    # Prepare a dictionary of results
    results = {}
    for image_id, details in json_data['test'].items():
        prev_id = details['prev']
        source_prev = image_id_to_path.get(prev_id, None)
        results[image_id] = {
            "prev": prev_id,
            "source_prev": source_prev
        }

    return results

# Function to save the results to a JSON file
def save_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Main function to execute
def main():
    json_file = 'image_match.json'  # Path to your JSON file
    txt_file = 'jpg_files_list.txt'  # Path to your txt file
    output_json ='image_match_source_test.json'  # Output JSON file

    # Read the JSON and txt files
    json_data = read_json(json_file)
    txt_data = read_txt(txt_file)

    # Find the sources of the prev image ids
    prev_sources = find_prev_image_source(json_data, txt_data)

    # Save the results to JSON
    save_to_json(prev_sources, output_json)

    print(f"Data saved to {output_json}")

if __name__ == '__main__':
    main()
