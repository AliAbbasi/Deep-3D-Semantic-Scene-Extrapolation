# create .npy file for each room of input json file

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

                # fetch the transformation matrix from node["transform"]
                transformation = np.asarray(node["transform"])
                transformation = transformation.reshape(4, 4)

                # find the node["modelId"] (is a string) from current directory
                object_voxel = np.load(str(node["modelId"] + ".npy"))

                # multiply transformation with object_voxel coordinates
                for x in range(object_voxel.shape[0]):
                    for y in range(object_voxel.shape[1]):
                        for z in range(object_voxel.shape[2]):
                            coordinate = np.ones((4, 1))
                            coordinate[0] = x
                            coordinate[1] = y
                            coordinate[2] = z

                            new_coordinate = transformation.dot(coordinate)
                            new_coordinate = map(int, new_coordinate)

                            # scene[new_coordinate[0], new_coordinate[1], new_coordinate[2]] = object_voxel[x, y, z]
                            scene[new_coordinate[0] + object_voxel.shape[0],
                                  new_coordinate[1] + object_voxel.shape[0],
                                  new_coordinate[2] + object_voxel.shape[0]] = object_voxel[x, y, z]

    np.save(str(json_file_input[:-5]) + ".npy", scene)
    # TODO:
    # put each voxelized object in the bbox of correspond place 

# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    # json to json s
    for json_file in glob.glob('*.json'):
        json_reader(json_file)
        # os.remove(json_file)

    # json to npy
    csv_loader()
    for json_file in glob.glob('*.json'):
        json_to_npy(json_file)
        # os.remove(json_file)
