# take each .json scene file and create correspond .npy file as voxelized scene

# ----------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------- 

def json_reader(json_file):
    rooms = []
    data = json.load(open(json_file)) 
    for level in data["levels"]: 
        level["id"]
        level["bbox"]
        level["nodes"] 
        # get floors here
        
        for node in level["nodes"]: 
            # get rooms here
            node["id"] # 0_0, 0_1, ...
            node["type"] # Object, Box, Ground, Room
            if node["type"] == "Object":
                # TODO:
                # get voxelized version of this object from related folder
                
# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    rooms_of_houses = []
    for json_file in glob.glob('*.json'):
        json_reader(json_file)
        # TODO:
        # delete the json_file after processing
