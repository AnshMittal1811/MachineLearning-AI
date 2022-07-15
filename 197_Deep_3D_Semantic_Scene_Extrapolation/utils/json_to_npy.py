# create .npy file for each room of input json file

# for subdir in *; do mv $subdir/house.json $subdir.json; done;
# find . -type d -name -delete
# find . -type f -size 0c -delete

# screen -X -S "session_name" quit
# screen -dmSL [session name] [commands]

# ----------------------------------------------------------------------------------

import json
import glob
import sys
import numpy as np
import csv
import math
import os
from multiprocessing import Pool

# ----------------------------------------------------------------------------------

model_category_mapping = []
coarse_grained_class = dict()

build_json_to_jsons = False
json_to_jsons_remove = False

build_json_to_npy = False
json_to_npy_remove = False

csv_load_flag = False
build_ply = True

batch_size_json_to_jsons = 50
batch_size_json_to_npy = 50

# ----------------------------------------------------------------------------------

def csv_loader():
    with open('meta_data/ModelCategoryMapping.csv') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        class_value = 0
        for row in dict_reader:
            model_category_mapping.append(row)
            key = row['coarse_grained_class']
            if key in coarse_grained_class:
                pass
            else:
                coarse_grained_class[key] = class_value
                class_value += 1


# ----------------------------------------------------------------------------------

def json_reader(json_file_input):
    data = json.load(open(json_file_input))
    for level in data["levels"]:
        for node in level["nodes"]:
            if node["type"] == "Room":
                get_room(node, json_file_input)

    if json_to_jsons_remove:
        os.remove(json_file_input)


# ----------------------------------------------------------------------------------

def get_room(room, json_file_input):
    room_id = room["id"]
    data = json.load(open(json_file_input))
    output_json = open('house/' + str(data["id"]) + "_" + str(room_id) + ".json", 'w')

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
                            # TODO: should be test in multi floor house
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

                if any(i < 0 for i in new_coordinate[0:3]) or \
                        any(i >= (input_object_voxel.shape[0] + max_dim * 3) for i in new_coordinate[0:3]):
                    pass
                else:
                    new_object_voxel[new_coordinate[0], new_coordinate[1], new_coordinate[2]] = \
                        input_object_voxel[x + int(max_dim / 2), y, z + int(max_dim / 2)]

    return new_object_voxel


# ----------------------------------------------------------------------------------

def json_to_npy(json_file_input):
    data = json.load(open(json_file_input))
    json_id = data["id"]
    glob_bbox_min = np.full(3, sys.maxint * 1.0)
    glob_bbox_max = np.full(3, -sys.maxint - 1 * 1.0)
    room_model_id = ""

    # to find the bbox_min and bbox_max of all objects
    for level in data["levels"]:
        for node in level["nodes"]:
            if node["type"] == "Room" and node["valid"] == 1:
                room_model_id = str(node["modelId"])
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
    scene_size = [i+5 for i in scene_size]
    if any(i < 0 for i in scene_size):
        if json_to_npy_remove:
            os.remove(json_file_input)
        return
    scene = np.zeros(scene_size)

    # put objects in their places
    for level in data["levels"]:
        for node in level["nodes"]:
            if node["type"] == "Object" and node["valid"] == 1:
                # fetch the transformation matrix from node["transform"]
                transformation = np.asarray(node["transform"]).reshape(4, 4)

                # find the node["modelId"] (is a string) from object directory
                object_voxel = np.load("object/" + str(node["modelId"] + ".npy"))

                bbox_min = np.asarray(node["bbox"]["min"])
                bbox_min -= glob_bbox_min
                bbox_min = map(int, (bbox_min * 100.0) / 6.0)

                # transformation
                object_voxel = trans_op(object_voxel, transformation)
                object_voxel = slice_non_zeroes(object_voxel)
                object_voxel = np.flip(object_voxel, 0)

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

                # give label to each voxel
                part_scene[np.where(object_voxel)] = object_voxel[np.where(object_voxel)]
                desired_label_value = 0
                for item in model_category_mapping:
                    if item['model_id'] == str(node["modelId"]):
                        desired_label_value = coarse_grained_class[item['coarse_grained_class']]
                        break
                part_scene[np.where(part_scene)] = desired_label_value
                scene[bbox_min[0]: bbox_min[0] + object_voxel.shape[0],
                      bbox_min[1]: bbox_min[1] + object_voxel.shape[1],
                      bbox_min[2]: bbox_min[2] + object_voxel.shape[2]] = part_scene

    # Add the walls, floor, ceiling to the scene
    for room in glob.glob('room/' + str(json_id) + '/' + '*.obj'):
        if str(room[38:-4]) == (room_model_id + 'w') or str(room[38:-4]) == (room_model_id + 'f'):
            desired_label_value = 1 if room[-5] == 'w' else 3
            vertices, faces = obj_reader(room)
            vertices -= glob_bbox_min
            vertices = (vertices * 100 / 6.0)
            for face in faces:
                if math.isnan(vertices[face[0]-1][0]) is False:
                    ver1 = map(int, vertices[face[0]-1])
                    ver2 = map(int, vertices[face[1]-1])
                    ver3 = map(int, vertices[face[2]-1])

                    min_coor = map(int, [min(ver1[0], ver2[0], ver3[0]), min(ver1[1], ver2[1], ver3[1]), min(ver1[2], ver2[2], ver3[2])])
                    max_coor = map(int, [max(ver1[0], ver2[0], ver3[0]), max(ver1[1], ver2[1], ver3[1]), max(ver1[2], ver2[2], ver3[2])])
                    min_coor = [0 if i < 0 else i for i in min_coor]
                    max_coor = [0 if i < 0 else i for i in max_coor]

                    max_coor[0] = max_coor[0] if max_coor[0] < scene.shape[0] else scene.shape[0]-1
                    max_coor[1] = max_coor[1] if max_coor[1] < scene.shape[1] else scene.shape[1]-1
                    max_coor[2] = max_coor[2] if max_coor[2] < scene.shape[2] else scene.shape[2]-1

                    if min_coor[0] == max_coor[0]:
                        scene[min_coor[0],
                              min_coor[1]: max_coor[1],
                              min_coor[2]: max_coor[2]] = desired_label_value
                    elif min_coor[1] == max_coor[1]:
                        scene[min_coor[0]: max_coor[0],
                              min_coor[1],
                              min_coor[2]: max_coor[2]] = desired_label_value
                    elif min_coor[2] == max_coor[2]:
                        scene[min_coor[0]: max_coor[0],
                              min_coor[1]: max_coor[1],
                              min_coor[2]] = desired_label_value
                    else:
                        scene[min_coor[0]: max_coor[0],
                              min_coor[1]: max_coor[1],
                              min_coor[2]: max_coor[2]] = desired_label_value

    np.save(str(json_file_input[:-5]) + ".npy", scene)

    if json_to_npy_remove:
        os.remove(json_file_input)

# ----------------------------------------------------------------------------------

def obj_reader(input_obj):
    vertices = []
    faces = []
    with open(input_obj, "r") as input_room:
        for line in input_room:
            if line[0:2] == "v ":
                vertices.append(line[2:])
            elif line[0:2] == "f ":
                faces.append(line[2:])

    for i in range(len(vertices)):
        vertices[i] = map(float, vertices[i].split())
    for i in range(len(faces)):
        faces[i] = faces[i].split()
        splitted = []
        for item in faces[i]:
            splitted.append(int(item.split("/")[0]))
        faces[i] = splitted
    vertices = np.asarray(vertices, dtype=float)

    return vertices, faces


# ----------------------------------------------------------------------------------

def npy_to_ply(input_npy_file):
    colors = [" 0 0 0 255  ", " 173 216 230 255", " 0 128 0 255", " 0 128 0 255", " 0 0 255 255", " 255 0 0 255",
              " 218 165 32 255", " 210 180 140 255", " 128 0   128 255", " 0  0 139 255", " 255 255 0 255",
              " 128 128 128 255", " 0 100 0 255", " 255 165 0 255", " 138 118 200 255 ",  " 236 206 244 255 ",
              " 126 172 209 255 ",  " 237 112 24 255  ",  " 158 197 220 255 ",  " 21 240 24 255   ",
              " 90 29 205 255  ",  " 183 246 66 255  ",  " 224 54 238 255  ",  " 39 129 50 255   ",
              " 252 204 171 255 ",  " 255 18 39 255   ",  " 118 76 69 255   ",  " 139 212 79 255  ",
              " 46 14 67 255    ",  " 142 113 129 255 ",  " 30 14 35 255    ",  " 17 90 54 255    ",
              " 125 89 247 255  ",  " 166 18 75 255   ",  " 129 142 18 255  ",  " 147 10 255 255  ",
              " 32 168 135 255  ",  " 245 199 6 255   ",  " 231 118 238 255 ",  " 84 35 213 255   ",
              " 214 230 80 255  ",  " 236 23 17 255   ",  " 92 207 229 255  ",  " 49 243 237 255  ",
              " 252 23 25 255   ",  " 209 224 126 255 ",  " 111 54 3 255    ",  " 96 11 79 255    ",
              " 169 56 226 255  ",  " 169 68 202 255  ",  " 107 32 121 255  ",  " 158 3 146 255   ",
              " 68 57 54 255    ",  " 212 200 217 255 ",  " 17 30 170 255   ",  " 254 162 238 255 ",
              " 16 120 52 255   ",  " 104 48 251 255  ",  " 176 49 253 255  ",  " 67 84 223 255   ",
              " 101 88 52 255   ",  " 204 50 193 255  ",  " 56 209 118 255  ",  " 79 74 216 255   ",
              " 104 142 255 255 ",  " 15 228 195 255  ",  " 185 168 157 255 ",  " 227 7 222 255   ",
              " 243 188 17 255  ",  " 20 85 135 255   ",  " 95 27 18 255    ",  " 189 126 21 255  ",
              " 69 254 247 255  ",  " 84 91 111 255   ",  " 8 153 222 255   ",  " 188 72 148 255  ",
              " 218 50 8 255    ",  " 183 217 27 255  ",  " 61 4 234 255    ",  " 31 113 81 255   ",
              " 75 130 78 255   ",  " 128 232 57 255  ",  " 16 183 77 255   ",  " 91 43 145 255   ",
              " 38 19 130 255   ",  " 64 236 113 255  ",  " 248 3 144 255   ",  " 194 157 62 255  ",
              " 143 219 101 255 ",  " 136 37 208 255  ",  " 102 144 241 255 ",  " 158 126 247 255 ",
              " 40 207 130 255  ",  " 88 131 224 255  ",  " 175 30 23 255   ",  " 42 224 197 255  ",
              " 23 175 34 255   ",  " 118 144 216 255 ",  " 32 128 149 255  ",  " 200 185 126 255 ",
              " 114 11 76 255   ",  " 28 60 36 255    ",  " 168 148 36 255  ",  " 57 246 83 255   "]

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
    ones = np.argwhere(input_np)
    if ones.size > 0:
        (x_start, y_start, z_start), (x_stop, y_stop, z_stop) = ones.min(0), ones.max(0) + 1
        return input_np[x_start:x_stop, y_start:y_stop, z_start:z_stop]
    else:
        return input_np


# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    # json to json s
    # counter = 1
    # if build_json_to_jsons:
    #     for json_file in glob.glob('house/*.json'):
    #         if len(str(json_file)) == 43:
    #             print "Counter: ", counter, str(json_file)
    #             json_reader(json_file)
    #             counter += 1

    index = 0
    p = Pool(batch_size_json_to_jsons)
    batch_arr = []
    counter = 0

    # json to json s
    if build_json_to_jsons:
        for json_file in glob.glob('house/*.json'):
            if len(str(json_file)) == 43:
                index += 1
                batch = []

                if counter < batch_size_json_to_jsons:
                    batch_arr.append(json_file)
                    counter += 1
                else:
                    counter = 0
                    batch.append(p.map(json_reader, batch_arr))
                    batch_arr = [json_file]
                    counter += 1
                    print index

        # one by one
        for json_file in batch_arr:
            json_reader(json_file)

    # load scenes information from meta files
    if csv_load_flag:
        csv_loader()

    # # json to npy
    # counter = 1
    # if build_json_to_npy:
    #     for json_file in glob.glob('house/*.json'):
    #         print "Counter: ", counter, str(json_file)
    #         json_to_npy(json_file)
    #         counter += 1

    index = 0
    p = Pool(batch_size_json_to_npy)
    batch_arr = []
    counter = 0

    # json to npy
    if build_json_to_npy:
        for json_file in glob.glob('house/*.json'):
            index += 1
            batch = []

            if counter < batch_size_json_to_npy:
                batch_arr.append(json_file)
                counter += 1
            else:
                counter = 0
                batch.append(p.map(json_to_npy, batch_arr))
                batch_arr = [json_file]
                counter += 1
                print index

        # one by one
        for json_file in batch_arr:
            json_to_npy(json_file)

    # npy to ply
    if build_ply:
        for npy_file in glob.glob('house/*.npy'):
            npy_to_ply(npy_file)
