#!/bin/bash

# Path to the JSON file
SCRIPT=visual_features_processing.py
json_file="train_chunk_1.json"

while IFS= read -r id; do
    commands+=("python ./${SCRIPT} --id $id")
done < <(jq -r 'keys[]' "$json_file")

# Run the commands in parallel using srun
printf "%s\n" "${commands[@]}"