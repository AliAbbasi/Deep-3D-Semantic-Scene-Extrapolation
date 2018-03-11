# create .npy file for each room of input json file

# ----------------------------------------------------------------------------------

import json
import glob
import sys
import os
import numpy as np
import csv
import math

# ----------------------------------------------------------------------------------

model_category_mapping = []
models = []
# scene = np.zeros((300, 300, 300))
build_ply = True
build_json_to_jsons = True


# ----------------------------------------------------------------------------------

def csv_loader():
    with open('meta_data/ModelCategoryMapping.csv') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        for row in dict_reader:
            model_category_mapping.append(row)

    with open('meta_data/models.csv') as csv_file:
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
            if level["id"].split("_")[0] == room_id.split("_")[0]:  # if room is in current level
                for node in level["nodes"]:
                    if node["type"] == "Room":
                        if node["id"] != room_id:
                            node["valid"] = 0
                    elif node["type"] == "Object":
                        if not int(node["id"].split("_")[1]) in node_indices:
                            # TODO: care about the object in the other levels
                               # or int(node["id"].split("_")[0]) != room_id.split("_")[0]:
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

def json_to_npy_with_trans(json_file_input):
    data = json.load(open(json_file_input))
    glob_bbox_min = np.full(3, sys.maxint * 1.0)
    glob_bbox_max = np.full(3, -sys.maxint - 1 * 1.0)

    # to find the bbox_min and bbox_max of all objects
    for level in data["levels"]:
        for node in level["nodes"]:
            if node["type"] == "Object" and node["valid"] == 1:
                bbox = node["bbox"]
                bbox_min = np.asarray(bbox["min"])
                bbox_max = np.asarray(bbox["max"])

                glob_bbox_min[0] = bbox_min[0] if bbox_min[0] < glob_bbox_min[0] else glob_bbox_min[0]
                glob_bbox_min[1] = bbox_min[1] if bbox_min[1] < glob_bbox_min[1] else glob_bbox_min[1]
                glob_bbox_min[2] = bbox_min[2] if bbox_min[2] < glob_bbox_min[2] else glob_bbox_min[2]

                glob_bbox_max[0] = bbox_max[0] if bbox_max[0] > glob_bbox_max[0] else glob_bbox_max[0]
                glob_bbox_max[1] = bbox_max[1] if bbox_max[1] > glob_bbox_max[1] else glob_bbox_max[1]
                glob_bbox_max[2] = bbox_max[2] if bbox_max[2] > glob_bbox_max[2] else glob_bbox_max[2]

    # put objects in their places
    for level in data["levels"]:
        for node in level["nodes"]:
            if node["type"] == "Object" and node["valid"] == 1:
                # fetch the transformation matrix from node["transform"]
                transformation = np.asarray(node["transform"])
                # transformation *= 100.0 / 6.0
                transformation = transformation.reshape(4, 4)

                # find the node["modelId"] (is a string) from current directory
                str_modelId = str(node["modelId"])
                object_voxel = np.load(str(node["modelId"] + ".npy"))

                bbox = node["bbox"]
                bbox_min = np.asarray(bbox["min"])
                bbox_max = np.asarray(bbox["max"])

                bbox_min -= glob_bbox_min
                bbox_max -= glob_bbox_min
                # TODO: care about the negative numbers in bbox
                bbox_min = map(int, (bbox_min * 100.0) / 6.0)
                bbox_max = map(int, (bbox_max * 100.0) / 6.0)

                for model in models:
                    if str(model["id"]) == str_modelId:
                        aligned_dims = model["aligned.dims"].split(",")

                aligned_dims = np.asarray(aligned_dims, dtype=float)
                aligned_dims /= 6.0
                max_dim = np.max(aligned_dims)

                for x in range(int(-max_dim / 2), int(max_dim / 2)):
                    for y in range(0, int(max_dim)):
                        for z in range(int(-max_dim / 2), int(max_dim / 2)):
                            coordinate = np.array([[x], [y], [z], [1]])
                            new_coordinate = transformation.dot(coordinate)
                            # new_coordinate *= 100.0 / 6.0
                            new_coordinate = map(int, new_coordinate)

                            if object_voxel[x + int(max_dim / 2), y, z + int(max_dim / 2)]:
                                scene[new_coordinate[0] + object_voxel.shape[0] + bbox_min[0],
                                      new_coordinate[1] + object_voxel.shape[0] + bbox_min[1],
                                      new_coordinate[2] + object_voxel.shape[0] + bbox_min[2]] = object_voxel[
                                    x + int(max_dim / 2), y, z + int(max_dim / 2)]

    np.save(str(json_file_input[:-5]) + ".npy", scene)


# ----------------------------------------------------------------------------------

def json_to_npy_no_trans(json_file_input):
    data = json.load(open(json_file_input))
    glob_bbox_min = np.full(3, sys.maxint * 1.0)
    glob_bbox_max = np.full(3, -sys.maxint - 1 * 1.0)

    # to find the bbox_min and bbox_max of all objects
    for level in data["levels"]:
        for node in level["nodes"]:
            if node["type"] == "Object" and node["valid"] == 1:
                bbox_min = np.asarray(node["bbox"]["min"])
                bbox_max = np.asarray(node["bbox"]["max"])

                glob_bbox_min[0] = bbox_min[0] if bbox_min[0] < glob_bbox_min[0] else glob_bbox_min[0]
                glob_bbox_min[1] = bbox_min[1] if bbox_min[1] < glob_bbox_min[1] else glob_bbox_min[1]
                glob_bbox_min[2] = bbox_min[2] if bbox_min[2] < glob_bbox_min[2] else glob_bbox_min[2]

                glob_bbox_max[0] = bbox_max[0] if bbox_max[0] > glob_bbox_max[0] else glob_bbox_max[0]
                glob_bbox_max[1] = bbox_max[1] if bbox_max[1] > glob_bbox_max[1] else glob_bbox_max[1]
                glob_bbox_max[2] = bbox_max[2] if bbox_max[2] > glob_bbox_max[2] else glob_bbox_max[2]

    # TODO: determine scene size with respect to the glob_bbox_max - glob_bbox_min
    scene_size = map(int, ((glob_bbox_max - glob_bbox_min) * 100.0) / 6.0)
    scene = np.zeros(scene_size)

    # put objects in their places
    for level in data["levels"]:
        for node in level["nodes"]:
            if node["type"] == "Object" and node["valid"] == 1:
                object_voxel = np.load("object/" + str(node["modelId"] + ".npy"))

                # get default object aligned dims
                for model in models:
                    if str(model["id"]) == str(node["modelId"]):
                        def_aligned_dims = np.around(np.asarray(map(float, model["aligned.dims"].split(","))))

                bbox_min = np.asarray(node["bbox"]["min"])
                bbox_max = np.asarray(node["bbox"]["max"])

                # get current object aligned dims
                cur_aligned_dims = np.around((bbox_max - bbox_min) * 100.0)

                # check if the object is in right direction or not
                if np.array_equal(def_aligned_dims, cur_aligned_dims):
                    pass
                elif def_aligned_dims[0] == cur_aligned_dims[2]:
                    object_voxel = np.transpose(object_voxel, (2, 1, 0))
                    object_voxel = np.flipud(object_voxel)

                if any(i < 0 for i in glob_bbox_min):
                    debbuger =1

                bbox_min -= glob_bbox_min
                bbox_max -= glob_bbox_min

                # TODO: care about the negative numbers in bbox
                bbox_min = map(int, (bbox_min * 100.0) / 6.0)
                bbox_max = map(int, (bbox_max * 100.0) / 6.0)

                # TODO: do matrix transpose with respect to the length of bbox in models csv and object_voxel bbox
                # TODO: what about diagonal objects
                # put object_voxel into scene where object_voxel = True
                part_scene = scene[bbox_min[0]: bbox_min[0] + object_voxel.shape[0],
                                   bbox_min[1]: bbox_min[1] + object_voxel.shape[0],
                                   bbox_min[2]: bbox_min[2] + object_voxel.shape[0]]
                # TODO: in some case the place of object is out of scene size,
                # TODO; in this case cut the object
                if part_scene.shape != object_voxel.shape:
                    debbugers= 1
                part_scene[np.where(object_voxel)] = object_voxel[np.where(object_voxel)]
                scene[bbox_min[0]: bbox_min[0] + object_voxel.shape[0],
                      bbox_min[1]: bbox_min[1] + object_voxel.shape[0],
                      bbox_min[2]: bbox_min[2] + object_voxel.shape[0]] = part_scene

        np.save(str(json_file_input[:-5]) + ".npy", scene)


# ----------------------------------------------------------------------------------

def npy_to_ply(input_npy_file):
    output_scene = np.load(input_npy_file)
    output = open(str(input_npy_file[:-4]) + ".ply", 'w')
    ply = ""
    ver_num = 0
    for idx1 in range(output_scene.shape[0]):
        for idx2 in range(output_scene.shape[1]):
            for idx3 in range(output_scene.shape[2]):
                if output_scene[idx1][idx2][idx3] >= 1:
                    ply = ply + str(idx1) + " " + str(idx2) + " " + str(idx3) + " 0 128 0 255" + "\n"
                    ver_num += 1
    output.write("ply" + "\n")
    output.write("format ascii 1.0" + "\n")
    output.write("comment VCGLIB generated" + "\n")
    output.write("element vertex " + str(ver_num) + "\n")
    output.write("property float x" + "\n")
    output.write("property float y" + "\n")
    output.write("property float z" + "\n")
    output.write("property uchar red" + "\n")
    output.write("property uchar green" + "\n")
    output.write("property uchar blue" + "\n")
    output.write("property uchar alpha" + "\n")
    output.write("element face 0" + "\n")
    output.write("property list uchar int vertex_indices" + "\n")
    output.write("end_header" + "\n")
    output.write(ply)
    output.close()
    print (str(input_npy_file[:-4]) + ".ply is Done.!")


# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    # json to json s
    if build_json_to_jsons:
        for json_file in glob.glob('*.json'):
            json_reader(json_file)
            # os.remove(json_file)

    # json to npy
    csv_loader()
    for json_file in glob.glob('*.json'):
        print (str(json_file))
        json_to_npy_no_trans(json_file)
        # os.remove(json_file)

    # TODO: give label to each voxel

    # npy to ply
    if build_ply:
        for npy_file in glob.glob('*.npy'):
            npy_to_ply(npy_file)
