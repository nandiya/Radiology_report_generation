import json

dict_progress_image = {}

data_stage1 = json.load(open("final_data.json", "r", encoding="utf-8"))

for i in data_stage1:
    temp = {}
    for j in data_stage1[i]:
        try:
            match_file = json.load(open("/group/pmc023/rnandiya/dataset/scene_graph/scene_graph/"+j+"_SceneGraph.json",
            "r", encoding="utf-8" ))
            progress_values = list(data_stage1[i][j]["progress"].values())
            temp[j] = {
                "prev": match_file["relationships"][0]["object_id"].split("_")[0],
                "progress": progress_values}
        except:
            pass
    dict_progress_image[i] = temp

with open("image_match.json", "w") as outfile: 
    json.dump(dict_progress_image, outfile)