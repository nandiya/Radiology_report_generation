import json
import os
import argparse
import pandas as pd

# Run: python data_prep.py \
#     --mimic_cxr_annotation /group/pmc023/rnandiya/mimic_annotation_all_a.json \
#         --progression_file all_progress.json \
#         --chexpert_file mimic-cxr-2.0.0-chexpert.csv 

parser = argparse.ArgumentParser()
parser.add_argument("--mimic_cxr_annotation", type=str)
parser.add_argument("--progression_file", type=str)
parser.add_argument("--chexpert_file", type=str)

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
j =0
for key in anno:
    temp_data={}
    j =0
    for i in anno[key]:
        #print(i)
        study_id = str(i["study_id"])
        subject_id = str(i["subject_id"])
        combination = subject_id+"_"+study_id
        report = i["report"]
        if "Findings" in report and "impression" in report:
            temp_data[i["id"]] = {"id": combination, "image_path": i["image_path"]}
            j = j+1
    splits[key] = temp_data


chexpert['patient_id'] = chexpert['subject_id'].astype(str) + '_' + chexpert['study_id'].astype(str)
chexpert['diseases_label'] = chexpert.drop(['subject_id', 'study_id', 'patient_id'], axis=1).values.tolist()
chexpert = chexpert[['patient_id', 'diseases_label']]

imageId_disease = {}
for data in splits:
    temp = {}
    for key in splits[data]: 
        patient_id = splits[data][key]["id"]
        if (chexpert['patient_id'] == patient_id).any():
            diseases_label = chexpert.loc[chexpert['patient_id'] == patient_id, 'diseases_label'].values[0]
            temp[key] = {"disease" : diseases_label, "image_path": splits[data][key]["image_path"]}
    imageId_disease[data]  = temp 


dict_progress = json.load(open(args.progression_file, "r", encoding="utf-8"))
train = {}
val = {}
test = {}


for i in imageId_disease["train"]:
    if i in dict_progress:
        train[i] = {"progress" : dict_progress[i], "disease": imageId_disease["train"][i]}
for i in imageId_disease["val"]:
    if i in dict_progress:
        val[i] = {"progress" : dict_progress[i], "disease": imageId_disease["val"][i]}
for i in imageId_disease["test"]:
    if i in dict_progress:
        test[i] = {"progress" : dict_progress[i], "disease": imageId_disease["test"][i]}

final_file = {
    "train": train,
    "val": val,
    "test" : test
}


with open("final_data.json", "w") as outfile: 
    json.dump(final_file, outfile)

