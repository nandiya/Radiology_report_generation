import json
import os
import argparse

# Run: python src_preprocessing/run_abn_preprocess.py \
    # --mimic_cxr_annotation data/mimic_cxr_annotation.json \
        # --progression_path /group/pmc023/rnandiya/dataset/scence_graph
        # --chexpert_file mimic-cxr-2.0.0-chexpert.csv\
            # --output_path data/mimic_abn_annotation_processed.json

parser = argparse.ArgumentParser()
parser.add_argument("--mimic_cxr_annotation", type=str)
parser.add_argument("--progression_path", type=str)
parser.add_argument("--chexpert_file", type=str)
parser.add_argument("--output_path", type=str)
args = parser.parse_args()

anno = json.load(open(args.mimic_cxr_annotation, "r", encoding="utf-8"))

img2progress = {}
img2diseases = {}
dict_progress = {}
train_id = []
for key in anno:
    if key == "train":
        train_id.append(key["train"]["id"])
    break
print(train_id)


# for filename in os.scandir(args.progression_path):
#     if filename.path.endswith(".json"):
#         length_data = length_data +1
#         f = open(filename)
#         data = json.load(f)
#         name = filename.name.split("_")[0]
#         progression_label = {"Better":0, "Worse" : 0, "Stable" :0}
#         if len(data["relationships"]) != 0 :
#            length_data_relation = length_data_relation + 1
#            for relation in data["relationships"]:
#                 if relation["predicate"].split("'")[1] == "Worse":
#                     progression_label["Worse"] = 1
#                 elif relation["predicate"].split("'")[1] == "Better" :
#                     progression_label["Better"] = 1
#                 else :
#                     progression_label["Stable"] = 1
#            dict_progress[name] = progression_label

