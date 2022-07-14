import sys
import subprocess
import json

import numpy as np

import os
import shutil

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove, path

from data.nuim_dynamics_dataloader import ICLNUIM_Dynamic_Dataset

def modify_pov_mesh(povray_icl_folder, icl_dataset):
    # load dynamics
    with open(sys.argv[3]) as f:
        dynamics = json.load(f)
        dynamics = dynamics[0]  # TODO SUPPORT MULTIPLE TRANSFORMATIONS IN ONE JSON

        dynamics["name"] = dynamics["name"][:len(dynamics["name"]) // 2] # in moved.txt the name is contained twice with a separating "_" in between, but in the mesh file it is not contained twice
        print(dynamics["name"])

        transformation = dynamics["transformation"]
        transformation = np.asarray(transformation).reshape((3, 4))
        transformation = icl_dataset.modify_dynamics_transformation(transformation)

        print(transformation)

    # Create temp file
    fh, abs_path = mkstemp()

    # construct path to original mesh file
    orig_mesh_file = path.join(povray_icl_folder, "living_room_POV_geom_original.inc")

    # construct path to actually used mesh file (the povray render script loads this file name)
    used_mesh_file = path.join(povray_icl_folder, "living_room_POV_geom.inc")

    # tells if we have found the place where to replace vertices in the file
    found = False
    skip_writing_range = None
    modified_vertex_lines = []

    with fdopen(fh, 'w') as new_file:
        with open(orig_mesh_file) as old_file:
            lines = old_file.readlines()
            for idx, line in enumerate(lines):
                if dynamics["name"] in line and not found:

                    print(f"Found starting point at line {idx}: {line}")

                    found = True
                    number_vertices = int(lines[idx+2][:-2]) # second next line is always in format '<vertices>,'
                    skip_writing_range = range(idx+3, idx+3+number_vertices) # these lines are all lines of the vertices in format '<x,y,z>'

                    print(f"Need to modify {number_vertices} vertices in line-range {skip_writing_range}")

                    # create the new vertex lines
                    for vertex_idx, vertex_line in enumerate(lines[skip_writing_range[0]:skip_writing_range[-1] + 1]):
                        print()
                        print(f"Convert {vertex_idx}-th line {vertex_line}")

                        if vertex_idx == len(skip_writing_range) - 1:
                            vertex_line = vertex_line[1:-2]  # remove leading '<' and trailing '>'
                        else:
                            vertex_line = vertex_line[1:-3] # remove leading '<' and trailing '>,'

                        print(f"Stripped line to {vertex_line}")

                        # parse vertex into numpy array
                        vertex = np.array([float(x) for x in vertex_line.split(",")])
                        vertex = np.append(vertex, 1) # add homogenous coordinate dimension

                        print(f"Converted line to vector {vertex}")

                        # transform vertex
                        vertex = transformation @ vertex

                        print(f"Transformed line to {vertex}")

                        # write modified line
                        if vertex_idx == len(skip_writing_range) - 1:
                            vertex_line = f'<{vertex[0]},{vertex[1]},{vertex[2]}>\n' # add leading '<' and trailing '>'
                        else:
                            vertex_line = f'<{vertex[0]},{vertex[1]},{vertex[2]}>,\n' # add leading '<' and trailing '>,'

                        modified_vertex_lines.append(vertex_line)
                        print(f"Modified line to {vertex_line}")

                # write this line: either from modified vertices or copy line
                if found and idx in skip_writing_range:
                    line = modified_vertex_lines[idx - skip_writing_range[0]]
                    print(f"Write modified vertex {line}")
                    new_file.write(line)
                else:
                    new_file.write(line)

    # Copy the file permissions from the old file to the new file
    copymode(used_mesh_file, abs_path)

    # Remove used mesh file
    remove(used_mesh_file)

    # Move new file to used mesh file
    move(abs_path, used_mesh_file)

if __name__ == '__main__':

    if len(sys.argv) != 3 and len(sys.argv) != 4:
        raise ValueError('Usage: ' + sys.argv[0] + ' <path_to_icl_with_camera_angles> <path_to_povray_icl_folder> (<path_to_moved.txt>)')

    remaining_seqs = ["seq0003"]

    for seq in remaining_seqs:
        print(sys.argv)
        sys.argv[1] = "/home/lukas/Desktop/datasets/ICL-NUIM/custom/" + seq
        sys.argv[3] = "/home/lukas/Desktop/datasets/ICL-NUIM/custom/" + seq + "/moved.txt"
        print(sys.argv)

        # ORIG
        d = ICLNUIM_Dynamic_Dataset(sys.argv[1])

        for i in range(d.__len__()):
            item = d.__getitem__(i)
            RT = item['cam']['RT1']
            RTinv = item['cam']['RT1inv']
            print(RT)

            bashCommand = f'povray +Iliving_room.pov +Oscene_00_{i:04d}.png +W640 +H480 ' \
                          f'Declare=val00={RT[0, 0]} Declare=val01={RT[1, 0]} Declare=val02={RT[2, 0]} ' \
                          f'Declare=val10={RT[0, 1]} Declare=val11={RT[1, 1]} Declare=val12={RT[2, 1]} ' \
                          f'Declare=val20={RT[0, 2]} Declare=val21={RT[1, 2]} Declare=val22={RT[2, 2]} ' \
                          f'Declare=val30={RT[0, 3]}  Declare=val31={RT[1, 3]} Declare=val32={RT[2, 3]} ' \
                          f'+FN16 +wt1 -d +L/usr/share/povray-3.7/include Declare=use_baking=2 -A'
            # +A0.0

            print(bashCommand)

            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd=sys.argv[2])
            output, error = process.communicate()

        for file in os.listdir(sys.argv[2]):
            if "scene" in file and ".png" in file:
                shutil.move(os.path.join(sys.argv[2], file), os.path.join(sys.argv[1], "original", file))

        # MOVED
        if len(sys.argv) == 4:
            d = ICLNUIM_Dynamic_Dataset(sys.argv[1])
            modify_pov_mesh(sys.argv[2], d)

            for i in range(d.__len__()):
                item = d.__getitem__(i)
                RT = item['cam']['RT1']
                RTinv = item['cam']['RT1inv']
                print(RT)

                bashCommand = f'povray +Iliving_room.pov +Oscene_00_{i:04d}.png +W640 +H480 ' \
                              f'Declare=val00={RT[0,0]} Declare=val01={RT[1,0]} Declare=val02={RT[2,0]} ' \
                              f'Declare=val10={RT[0,1]} Declare=val11={RT[1,1]} Declare=val12={RT[2,1]} ' \
                              f'Declare=val20={RT[0,2]} Declare=val21={RT[1,2]} Declare=val22={RT[2,2]} ' \
                              f'Declare=val30={RT[0,3]}  Declare=val31={RT[1,3]} Declare=val32={RT[2,3]} ' \
                              f'+FN16 +wt1 -d +L/usr/share/povray-3.7/include Declare=use_baking=2 -A'
                # +A0.0

                print(bashCommand)

                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd=sys.argv[2])
                output, error = process.communicate()

            for file in os.listdir(sys.argv[2]):
                if "scene" in file and ".png" in file:
                    shutil.move(os.path.join(sys.argv[2], file), os.path.join(sys.argv[1], "moved", file))

