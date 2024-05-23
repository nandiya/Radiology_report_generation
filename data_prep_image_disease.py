import json
import os
import argparse
import pandas as pd

chexpert = pd.read_csv("mimic-cxr-2.0.0-chexpert.csv")
chexpert = chexpert.fillna(0)


anno = json.load(open("/group/pmc023/rnandiya/dataset/mimic_annotation_all.json", "r", encoding="utf-8"))
splits = {}

for key in anno:
    temp_data={}
    for i in anno[key]:
        study_id = str(i["study_id"])
        subject_id = str(i["subject_id"])
        combination = subject_id+"_"+study_id
        temp_data[i["id"]] = combination
    splits[key] = temp_data


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

with open("imageId_disease.json", "w") as outfile: 
    json.dump(imageId_disease, outfile)