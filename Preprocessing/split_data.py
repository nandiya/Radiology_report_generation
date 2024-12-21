import json
import os


imageId_disease = json.load(open("final_data.json", "r", encoding="utf-8"))

def split_data(data, chunk_size):
    items = list(data.items())
    return [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]


chunks = split_data(imageId_disease["train"], 5400)


for i, chunk in enumerate(chunks):
    with open(f'train_chunk_{i + 1}.json', 'w') as f:
        json.dump(chunk, f, indent=4)

print("JSON files created successfully.")
print("JSON files created successfully.")