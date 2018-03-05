# create .npy voxelized file from each object.obj

# ----------------------------------------------------------------------------------

import glob
import os
import csv
import math

# ----------------------------------------------------------------------------------

model_category_mapping = []
models = []
voxel_size = 6  # cm

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

def compute_size(input_obj_file):
    file_name = str(input_obj_file)[0:-4]
    min_point = []
    max_point = []

    for model in models:
        if str(model["id"]) == file_name:
            min_point = model["minPoint"].split(",")
            max_point = model["maxPoint"].split(",")

    # map from str to int
    if min_point and max_point:
        min_point = map(float, min_point)
        max_point = map(float, max_point)

    # find the longest dimension
    x_dim = math.sqrt((min_point[0] - max_point[0]) ** 2)
    y_dim = math.sqrt((min_point[1] - max_point[1]) ** 2)
    z_dim = math.sqrt((min_point[2] - max_point[2]) ** 2)
    long_dim = max(x_dim, y_dim, z_dim)
    return int((long_dim * 100) / voxel_size)

# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    csv_loader()
    rooms_of_houses = []
    for obj_file in glob.glob('*.obj'):
        size = compute_size(obj_file)
        os.system("binvox.exe -d " + str(size) + " " + str(obj_file))
        # os.remove(obj_file)

    for binvox_file in glob.glob('*.binvox'):
        pass
        # TODO: do binvox_to_npy