import json
import os
import argparse
import pandas as pd


dict_progress = {}

for filename in os.scandir("/group/pmc023/rnandiya/dataset/scene_graph/scene_graph"):
    if filename.path.endswith(".json"):
        f = open(filename)
        data = json.load(f)
        name = filename.name.split("_")[0]
        progression_label = {"Better":0, "Worse" : 0, "Stable" :0}
        for relation in data["relationships"]:
            if relation["predicate"].split("'")[1] == "Worse":
                progression_label["Worse"] = 1
            if relation["predicate"].split("'")[1] == "Better" :
                progression_label["Better"] = 1
            if relation["predicate"].split("'")[1] == "No status change" :
                progression_label["Stable"] = 1
        dict_progress[name] = progression_label

print(len(dict_progress))
with open("all_progress.json", "w") as outfile: 
    json.dump(dict_progress, outfile)