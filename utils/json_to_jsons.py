# create specific json file for each room of input json file

# ----------------------------------------------------------------------------------

import json 
import glob
import sys
import os

# ----------------------------------------------------------------------------------

def json_reader(json_file):
    data = json.load(open(json_file)) 
    for level in data["levels"]:
        for node in level["nodes"]:
            if node["type"] == "Room":
                get_room(node, json_file) 
                
# ----------------------------------------------------------------------------------

def get_room(room, json_file):
    room_id = room["id"]   
    data = json.load(open(json_file))  
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

if __name__ == '__main__':

    rooms_of_houses = []
    for json_file in glob.glob('*.json'):
        json_reader(json_file)
        # os.remove(json_file)
