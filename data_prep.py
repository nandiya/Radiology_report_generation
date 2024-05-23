import json
import os
import argparse
import pandas as pd

# Run: python data_prep.py \
#     --mimic_cxr_annotation /group/pmc023/rnandiya/dataset/mimic_annotation_all.json \
#         --progression_path /group/pmc023/rnandiya/dataset/scence_graph \
#         --chexpert_file mimic-cxr-2.0.0-chexpert.csv \
#             --output_path data/mimic_abn_annotation_processed.json

parser = argparse.ArgumentParser()
parser.add_argument("--mimic_cxr_annotation", type=str)
parser.add_argument("--progression_path", type=str)
parser.add_argument("--chexpert_file", type=str)
parser.add_argument("--output_path", type=str)
args = parser.parse_args()

anno = json.load(open(args.mimic_cxr_annotation, "r", encoding="utf-8"))
chexpert = pd.read_csv(args.chexpert_file)
chexpert = chexpert.fillna(0)

img2progress = {}
img2diseases = {}
dict_progress = {}
observation = []
val_id = []
splits = {}

for key in anno:
    temp_data={}
    for i in anno[key]:
        study_id = str(i["study_id"])
        subject_id = str(i["subject_id"])
        combination = subject_id+"_"+study_id
        temp_data[i["id"]] = combination
    splits[key] = temp_data

# with open("all_progress.json", "w") as outfile: 
#     json.dump(splits, outfile)

patient_disease = {}
chexpert['patient_id'] = chexpert['subject_id'].astype(str) + '_' + chexpert['study_id'].astype(str)
chexpert['diseases_label'] = chexpert.drop(['subject_id', 'study_id', 'patient_id'], axis=1).values.tolist()
chexpert = chexpert[['patient_id', 'diseases_label']]

imageId_disease = {}
for data in splits:
    temp = {}
    for key in splits[data]: 
        patient_id = splits[data][key]
        if (chexpert['patient_id'] == patient_id).any():
            diseases_label = chexpert.loc[chexpert['patient_id'] == patient_id, 'diseases_label'].values[0]
            temp[key] = diseases_label
    imageId_disease[data]  = temp 

    
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

f = open("all_progress.json")
dict_progress = json.load(f)
train = {}
val = {}
test = {}
for i in dict_progress:
    if i in imageId_disease["train"]:
        train[i] = dict_progress[i]
    elif i in imageId_disease["val"]:
        val[i] = dict_progress[i]
    else:
        test[i] = dict_progress[i]
    break


print(len(train))
print(len(val))
print(len(test))

better_count = 0
worse_count = 0
stable_count = 0

for key, nested_dict in train.items():
    print(key)
    print(nested_dict)
    # Sum up the counts for each category
    better_count += nested_dict.get("better", 0)
    worse_count += nested_dict.get("worse", 0)
    stable_count += nested_dict.get("stable", 0)


print("Better:", better_count)
print("Worse:", worse_count)
print("Stable:", stable_count)
