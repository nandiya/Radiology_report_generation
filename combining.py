import os
import json
from glob import glob

def combine_json_files(output_file="combined_list_regions_train.json"):
    combined_data = {}

    # Find all JSON files that start with 'list_regions' in the current directory
    json_files = glob('regions*.json')

    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            combined_data.update(data)

    # Output the combined data to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)

if __name__ == "__main__":
    combine_json_files()
