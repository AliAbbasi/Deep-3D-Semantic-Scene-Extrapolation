# take each .json scene file and create correspond .npy file as voxelized scene

# ----------------------------------------------------------------------------------

import csv
import glob
import json
import sys

# ---------------------------------------------------------------------------------- 

model_category_mapping = {}
models = {}

# ---------------------------------------------------------------------------------- 

def csv_loader():
    with open('ModelCategoryMapping.csv') as csvfile:
        model_category_mapping = csv.DictReader(csvfile)
        
    with open('models.csv') as csvfile:
        models = csv.DictReader(csvfile) 
        for row in models:
            print (type(row))
            for key, value in row.items() :
                print (key, value)
            sys.exit(0)
            print(row['minPoint'])

    
    
    print (models )
    print (models[0] )
    sys.exit(0)
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
                pass
                # TODO:
                # get voxelized version of this object from related folder
                
# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    csv_loader()
    rooms_of_houses = []
    for json_file in glob.glob('*.json'):
        json_reader(json_file)
        # TODO:
        # delete the json_file after processing
