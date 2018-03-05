# take each .json scene file and create correspond .npy file as voxelized scene

# ----------------------------------------------------------------------------------

import csv
import glob
import json
import sys

# ---------------------------------------------------------------------------------- 

model_category_mapping = []
models = []

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

    print (models)
    print (models[0])
    sys.exit(0)
# ---------------------------------------------------------------------------------- 

def json_reader(json_file_input):
    data = json.load(open(json_file_input))
    for level in data["levels"]:
        for node in level["nodes"]:
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
