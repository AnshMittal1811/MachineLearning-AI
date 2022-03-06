from sys import argv
from utils import *

# NOTE: Import this file from blender python console

# Args for the program
INPUT_PATH = "/path/to/_semantic.ply"
OUTPUT_PATH = "/path/to/new_semantic.ply"

blender_init()

#####################################
# Available operations and examples:#
#####################################
# File has to be parsed again manually in the beginning because blender doesn't read vertex colors and object IDs. 
# Make sure to use the same file as read by blender.
# importer(INPUT_PATH)

# Select faces from UI first then (make sure to use face selection, i.e. not edge, not vertex):
# get_objID_of_selection()

# select_faces_of_obj(17, sel_extend=False)

# cut_object(17)

# Transformations via console:
# R = rotate_selection(45, 'X', in_degrees=True)
# T = translate_selection((1, 0, -1))

# Alternatively, transformations via UI: Move (bpy.ops.transform.translate(value=(x,y,z))) & Rotate (bpy.ops.transform.rotate(value=radians, orient_axis=axis,..)
# Check the logs and by passing "value" and "axis" fields into respective args in get_RT_matrix, one can get overall RT matrix:
# RT = get_RT_matrix((30, 0, 0), (1, 0, -1), in_degrees=True), rotation vector (30, 0, 0) implies rotation is around "X" axis with 30 degrees

# Used internally to translate center of selected faces to origin during rotation. However, it can be useful in debugging as well.
# center = selection_center()

# Perform multiple translate_selection rotate_selection calls and multiply matrices in correct order. 
# Then, call this function with inverted composite transformation. It's a good debug tool.
# transform_selection(transform)

# Compose all transformations performed up to this call. 
# "transforms" list automatically keeps them in the correct order. 
# Set clear flag to empty the list.
# compose_transforms(transforms, clear=True):

# Pass a single (composite) transform matrix or a list of transform matrices (in this case, it will be composed automatically)
# Exports moved.txt file containing transformation matrix and ID of the object (here: 12) that was transformed.
# export_moved_info(path, transforms, 12, clear=True):

# Finally, export modified mesh with:
# exporter(OUTPUT_PATH)

