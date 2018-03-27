# create .npy file for each room of input json file

# ----------------------------------------------------------------------------------

import json
import glob
import sys
import os
import numpy as np
import csv
import random

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

def trans_op(input_object_voxel, input_transformation):
    # TODO: there is a problem with trans, after trans some part of objects are missing, i.e., 238
    max_dim = np.max(input_object_voxel.shape)
    new_object_voxel = np.zeros((input_object_voxel.shape[0] + max_dim * 3,
                                 input_object_voxel.shape[1] + max_dim * 3,
                                 input_object_voxel.shape[2] + max_dim * 3))

    for x in range(int(-max_dim / 2), int(max_dim / 2)):
        for y in range(0, int(max_dim)):
            for z in range(int(-max_dim / 2), int(max_dim / 2)):
                coordinate = np.array([[x], [y], [z], [1]])
                new_coordinate = input_transformation.dot(coordinate)
                new_coordinate += max_dim + int(max_dim / 2) + 1
                new_coordinate = np.around(new_coordinate)
                new_coordinate = np.asarray(map(int, new_coordinate))

                # TODO: there is two problems, 1: the new_coors are negative, 2: new_coors are larger than expected
                if any(i < 0 for i in new_coordinate[0:3]) or any(i > (input_object_voxel.shape[0] + max_dim * 3) for i in new_coordinate[0:3]):
                    pass
                else:
                    new_object_voxel[new_coordinate[0], new_coordinate[1], new_coordinate[2]] = \
                        input_object_voxel[x + int(max_dim/2), y, z + int(max_dim/2)]

    # for x in range(input_object_voxel.shape[0]):
    #     for y in range(input_object_voxel.shape[1]):
    #         for z in range(input_object_voxel.shape[2]):
    #             coordinate = np.array([[x], [y], [z], [1]])
    #             new_coordinate = input_transformation.dot(coordinate)
    #             new_coordinate = np.asarray(map(int, np.around(new_coordinate)))
    #             int_max_dim = int(max_dim / 2.0)
    #             new_coordinate += int_max_dim
    #             new_object_voxel[new_coordinate[0], new_coordinate[1], new_coordinate[2]] = input_object_voxel[x, y, z]

    return new_object_voxel


# ----------------------------------------------------------------------------------

def json_to_npy(json_file_input):
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

    # determine scene size with respect to the glob_bbox_max - glob_bbox_min
    scene_size = map(int, ((glob_bbox_max - glob_bbox_min) * 100.0) / 6.0)
    scene = np.zeros(scene_size)

    # put objects in their places
    for level in data["levels"]:
        for node in level["nodes"]:
            if node["type"] == "Object" and node["valid"] == 1:
                # fetch the transformation matrix from node["transform"]
                transformation = np.asarray(node["transform"]).reshape(4, 4)

                # ----------------------------------------
                # TODO: determine if the transformation is diagonal, then keep it, if it is axis align, then convert it to int
                # but how ???
                # TODO: I think we need float trans matrix for diagonal transformation
                # TODO: So convert float to int means that there is no diagonal transformation
                # TODO: Maybe filling the empty spaces between voxels after transformation is better solution
                # transformation = np.around(transformation)
                # transformation = transformation.astype(np.int64)
                # ----------------------------------------

                # find the node["modelId"] (is a string) from current directory
                str_modelId = str(node["modelId"])
                object_voxel = np.load("object/" + str(node["modelId"] + ".npy"))

                bbox_min = np.asarray(node["bbox"]["min"])
                bbox_max = np.asarray(node["bbox"]["max"])

                # TODO: cur_aligned_dims for what ???

                bbox_min -= glob_bbox_min
                bbox_max -= glob_bbox_min

                # TODO: care about the negative numbers in bbox
                bbox_min = map(int, (bbox_min * 100.0) / 6.0)

                # transformation
                object_voxel = trans_op(object_voxel, transformation)
                object_voxel = slice_non_zeroes(object_voxel)
                object_voxel = np.flip(object_voxel, 0)

                # ==================================================
                # output = open(str(str_modelId) + ".ply", 'w')
                # ply = ""
                # ver_num = 0
                # for idx1 in range(object_voxel.shape[0]):
                #     for idx2 in range(object_voxel.shape[1]):
                #         for idx3 in range(object_voxel.shape[2]):
                #             if object_voxel[idx1][idx2][idx3] >= 1:
                #                 ply = ply + str(idx1) + " " + str(idx2) + " " + str(idx3) + " 0 128 0 255" + "\n"
                #                 ver_num += 1
                # output.write("ply" + "\n")
                # output.write("format ascii 1.0" + "\n")
                # output.write("comment VCGLIB generated" + "\n")
                # output.write("element vertex " + str(ver_num+8) + "\n")
                # output.write("property float x" + "\n")
                # output.write("property float y" + "\n")
                # output.write("property float z" + "\n")
                # output.write("property uchar red" + "\n")
                # output.write("property uchar green" + "\n")
                # output.write("property uchar blue" + "\n")
                # output.write("property uchar alpha" + "\n")
                # output.write("element face 0" + "\n")
                # output.write("property list uchar int vertex_indices" + "\n")
                # output.write("end_header" + "\n")
                # output.write(ply)
                # output.write("0 0 0 0 128 0 255 \n")
                # output.write("0 0 "+str(object_voxel.shape[2])+" 0 128 0 255 \n")
                # output.write("0 "+str(object_voxel.shape[1])+" 0 0 128 0 255 \n")
                # output.write(str(object_voxel.shape[0])+" 0 0 0 128 0 255 \n")
                # output.write(str(object_voxel.shape[0])+" " +str(object_voxel.shape[1])+" "+str(object_voxel.shape[2])+" 0 128 0 255 \n")
                # output.write("0 " +str(object_voxel.shape[1])+" "+str(object_voxel.shape[2])+" 0 128 0 255 \n")
                # output.write(str(object_voxel.shape[0])+" 0 "+ str(object_voxel.shape[2])+" 0 128 0 255 \n")
                # output.write(str(object_voxel.shape[0])+" " +str(object_voxel.shape[1])+" 0 0 128 0 255 \n")
                # output.close()
                # print (str(str_modelId) + ".ply is Done.!")
                # ==================================================

                # put object_voxel into scene where object_voxel = True
                part_scene = scene[bbox_min[0]: bbox_min[0] + object_voxel.shape[0],
                                   bbox_min[1]: bbox_min[1] + object_voxel.shape[1],
                                   bbox_min[2]: bbox_min[2] + object_voxel.shape[2]]
                # in some case the place of object is out of scene size, cut the object to fit
                if part_scene.shape != object_voxel.shape:
                    object_voxel = object_voxel[:part_scene.shape[0], :part_scene.shape[1], :part_scene.shape[2]]
                    part_scene = scene[bbox_min[0]: bbox_min[0] + object_voxel.shape[0],
                                       bbox_min[1]: bbox_min[1] + object_voxel.shape[1],
                                       bbox_min[2]: bbox_min[2] + object_voxel.shape[2]]

                # TODO: give label to each voxel
                # random color to each voxel, TODO: fix it later
                random_color = random.randint(1, 13)
                part_scene[np.where(object_voxel)] = object_voxel[np.where(object_voxel)]
                part_scene[np.where(part_scene)] = random_color
                scene[bbox_min[0]: bbox_min[0] + object_voxel.shape[0],
                      bbox_min[1]: bbox_min[1] + object_voxel.shape[1],
                      bbox_min[2]: bbox_min[2] + object_voxel.shape[2]] = part_scene

        # TODO; before save the scene, put the walls, floor and ceiling
        np.save(str(json_file_input[:-5]) + ".npy", scene)


# ----------------------------------------------------------------------------------

def npy_to_ply(input_npy_file):
    colors = []
    colors.append(" 0 0 0 255  ")  # balck      for 0  'empty'
    colors.append(" 139 0 0 255")  # dark red   for 1  'ceiling'
    colors.append(" 0 128 0 255")  # green      for 2  'floor'
    colors.append(" 173 216 230 255")  # light blue for 3  'wall'
    colors.append(" 0 0 255 255")  # blue       for 4  'window'
    colors.append(" 255 0 0 255")  # red        for 5  'door'
    colors.append(" 218 165 32 255")  # goldenrod  for 6  'chair'
    colors.append(" 210 180 140 255")  # tan        for 7  'bed'
    colors.append(" 128 0   128 255")  # purple     for 8  'sofa'
    colors.append(" 0  0 139 255")  # dark blue  for 9  'table'
    colors.append(" 255 255 0 255")  # yellow     for 10 'coffe table'
    colors.append(" 128 128 128 255")  # gray       for 11 'shelves'
    colors.append(" 0 100 0 255")  # dark green for 12 ' '
    colors.append(" 255 165 0 255")  # orange     for 13 'furniture'

    output_scene = np.load(input_npy_file)
    output = open(str(input_npy_file[:-4]) + ".ply", 'w')
    ply = ""
    ver_num = 0
    for idx1 in range(output_scene.shape[0]):
        for idx2 in range(output_scene.shape[1]):
            for idx3 in range(output_scene.shape[2]):
                if output_scene[idx1][idx2][idx3] >= 1:
                    ply = ply + str(idx1) + " " + str(idx2) + " " + str(idx3) + str(
                        colors[int(output_scene[idx1][idx2][idx3])]) + "\n"
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

def slice_non_zeroes(input_np):
    # TODO: is this if in right way ???
    ones = np.argwhere(input_np)
    if ones.size > 0:
        (x_start, y_start, z_start), (x_stop, y_stop, z_stop) = ones.min(0), ones.max(0) + 1
        return input_np[x_start:x_stop, y_start:y_stop, z_start:z_stop]
    else:
        return input_np


# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    # json to json s
    if build_json_to_jsons:
        for json_file in glob.glob('*.json'):
            json_reader(json_file)
            os.remove(json_file)

    # json to npy
    csv_loader()
    for json_file in glob.glob('*.json'):
        print (str(json_file))
        json_to_npy(json_file)
        # os.remove(json_file)

    # npy to ply
    if build_ply:
        for npy_file in glob.glob('*.npy'):
            npy_to_ply(npy_file)

            