# create specific json file for each room of input json file

# ----------------------------------------------------------------------------------

import json 
import glob
import sys
import os
import numpy as np
import csv

# ----------------------------------------------------------------------------------

model_category_mapping = []
models = []
scene = np.zeros((200, 200, 100))

# ----------------------------------------------------------------------------------

def csv_loader():
    with open('ModelCategoryMapping.csv') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        for row in dict_reader:
            model_category_mapping.append(row)

    with open('models.csv') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        for row in dict_reader:
            models.append(row)

# ----------------------------------------------------------------------------------

def json_reader(input_json_file):
    data = json.load(open(input_json_file))
    for level in data["levels"]:
        for node in level["nodes"]:
            if node["type"] == "Room":
                get_room(node, input_json_file)
                
# ----------------------------------------------------------------------------------

def get_room(room, input_json_file):
    room_id = room["id"]   
    data = json.load(open(input_json_file))
    output_json = open(str(data["id"]) + "_" + str(room_id) + ".json", 'w') 
    
    if "nodeIndices" in room:
        node_indices = room["nodeIndices"]
        
        for level in data["levels"]: 
            if level["id"].split("_")[0] == room_id.split("_")[0]:
                for node in level["nodes"]:
                    if node["type"] == "Room":
                        if node["id"] != room_id:
                            node["valid"] = 0
                    elif node["type"] == "Object":
                        if not int(node["id"].split("_")[1]) in node_indices:
                            node["valid"] = 0
                    elif node["type"] == "Ground":
                        node["valid"] = 0
                    else:  # Box
                        node["valid"] = 0
            else:
                for node in level["nodes"]:
                    node["valid"] = 0
                    
        json.dump(data, output_json)

# ----------------------------------------------------------------------------------

def json_to_npy(json_file_input):
    data = json.load(open(json_file_input))
    for level in data["levels"]:
        for node in level["nodes"]:
            if node["type"] == "Object" and node["valid"] == 1:
                pass
                # TODO:
                # get node["modelId"] bbox from 'models' dictionary
                # get voxelized version of this id from related folder
                # we should multiply the object with its transformation matrix (how?)

# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    # create json files for each room
    for json_file in glob.glob('*.json'):
        json_reader(json_file)
        # os.remove(json_file)

    # create .npy scene file for each .json
    csv_loader()
    for json_file in glob.glob('*.json'):
        json_to_npy(json_file)
        # os.remove(json_file)
