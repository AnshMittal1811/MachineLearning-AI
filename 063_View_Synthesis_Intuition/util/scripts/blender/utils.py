import bpy
import bmesh
from collections import Counter
import json
import struct
import mathutils
import re
import numpy as np
from scipy.spatial.transform import Rotation

HEADER = \
"ply\n" + \
"format binary_little_endian 1.0\n" + \
"comment Written with hapPLY (https://github.com/nmwsharp/happly)\n" + \
"element vertex {vcount}\n" + \
"property float x\n" + \
"property float y\n" + \
"property float z\n" + \
"property uchar red\n" + \
"property uchar green\n" + \
"property uchar blue\n" + \
"element face {fcount}\n" + \
"property list uchar int vertex_indices\n" + \
"property int object_id\n" + \
"end_header\n"

REGION_OFFSET = 1000000
seg_rxp = re.compile(r"E  \d+ (\d+) (\d+) .*")
idx_rxp = re.compile(r".*region(\d+).ply")

SEG_COLORS = [[0.12156863, 0.46666667, 0.70588235],
              [0.68235294, 0.78039216, 0.90980392],
              [1.        , 0.49803922, 0.05490196],
              [1.        , 0.73333333, 0.47058824],
              [0.17254902, 0.62745098, 0.17254902],
              [0.59607843, 0.8745098 , 0.54117647],
              [0.83921569, 0.15294118, 0.15686275],
              [1.        , 0.59607843, 0.58823529],
              [0.58039216, 0.40392157, 0.74117647],
              [0.77254902, 0.69019608, 0.83529412],
              [0.54901961, 0.3372549 , 0.29411765],
              [0.76862745, 0.61176471, 0.58039216],
              [0.89019608, 0.46666667, 0.76078431],
              [0.96862745, 0.71372549, 0.82352941],
              [0.49803922, 0.49803922, 0.49803922],
              [0.78039216, 0.78039216, 0.78039216],
              [0.7372549 , 0.74117647, 0.13333333],
              [0.85882353, 0.85882353, 0.55294118],
              [0.09019608, 0.74509804, 0.81176471],
              [0.61960784, 0.85490196, 0.89803922],
              [0.22352941, 0.23137255, 0.4745098 ],
              [0.32156863, 0.32941176, 0.63921569],
              [0.41960784, 0.43137255, 0.81176471],
              [0.61176471, 0.61960784, 0.87058824],
              [0.38823529, 0.4745098 , 0.22352941],
              [0.54901961, 0.63529412, 0.32156863],
              [0.70980392, 0.81176471, 0.41960784],
              [0.80784314, 0.85882353, 0.61176471],
              [0.54901961, 0.42745098, 0.19215686],
              [0.74117647, 0.61960784, 0.22352941],
              [0.90588235, 0.72941176, 0.32156863],
              [0.90588235, 0.79607843, 0.58039216],
              [0.51764706, 0.23529412, 0.22352941],
              [0.67843137, 0.28627451, 0.29019608],
              [0.83921569, 0.38039216, 0.41960784],
              [0.90588235, 0.58823529, 0.61176471],
              [0.48235294, 0.25490196, 0.45098039],
              [0.64705882, 0.31764706, 0.58039216],
              [0.80784314, 0.42745098, 0.74117647],
              [0.87058824, 0.61960784, 0.83921569]]

# Global variables used frequently in functions
obj = obj_data = mesh = VERTEX_COUNT = FACE_COUNT = None

vertex_to_rgb = []
face_to_objID = []
objID_to_face = {}
seg_to_objID = {}

# List of transformation matrices stored when translate_selection, rotate_selection or transform_selection are called.
# Later transformation matrices can be composed with compose_transforms function
transforms = []

rotx90minus = np.array([[1.0,  0.0, 0.0, 0.0], 
                        [0.0,  0.0, 1.0, 0.0], 
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0,  0.0, 0.0, 1.0]])
rotx90plus = np.linalg.inv(rotx90minus)

def reset_state():
  global obj, obj_data, mesh, VERTEX_COUNT,FACE_COUNT, vertex_to_rgb,face_to_objID,objID_to_face, seg_to_objID, transforms
  obj = obj_data = mesh = VERTEX_COUNT = FACE_COUNT = None
  vertex_to_rgb.clear()
  face_to_objID.clear()
  objID_to_face.clear()
  seg_to_objID.clear()
  transforms.clear()

def blender_init():
    """
    Sets global variables that are required for scene manipulation. 
    Undoing an action may delete existing global variables. If that's the case, call this method again.
    """

    global obj, obj_data, mesh, VERTEX_COUNT, FACE_COUNT, transforms
    
    # Retrieve data from blender
    obj = bpy.context.object              # Get mesh in the scene
    bpy.ops.object.mode_set(mode='EDIT')  # Put blender into edit mode
    obj_data = obj.data
    mesh = bmesh.from_edit_mesh(obj.data) # Access faces and vertices of obj

    # If index error happens due to outdated internal index table, use the followings:
    # mesh.verts.ensure_lookup_table()
    # mesh.faces.ensure_lookup_table()
    # mesh.edges.ensure_lookup_table()

    VERTEX_COUNT = len(mesh.verts)
    FACE_COUNT = len(mesh.faces)

def importer(path):
    global vertex_to_rgb, face_to_objID, objID_to_face
    
    vertex_to_rgb.clear()
    face_to_objID.clear()
    objID_to_face.clear()

    # Read PLY file
    data = None
    with open(path, "rb") as file:
        data = file.read()

    is_binary = data.find(b"ascii", 0, 20) == -1
    
    # Binary PLY
    if is_binary:
        print("Parsing binary file...")
        
        # Count number of lines for header (may vary)
        HEADER_SIZE = 0
        VERTEX_COUNT = None
        FACE_COUNT = None
        
        line = b""
        start = 0
        max = len(data)
        while line.lower() != "end_header":
            end = data.find(b"\n", start, max)
            line = data[start:end].decode("ascii")
            if "element vertex" in line:
                VERTEX_COUNT = int(line.split(" ")[-1])
            elif "element face" in line:
                FACE_COUNT = int(line.split(" ")[-1])
                
            HEADER_SIZE += end - start + 1
            start = end + 1
 
        print("HEADER_SIZE (in bytes):", HEADER_SIZE)
        print("VERTEX_COUNT:", VERTEX_COUNT)
        print("FACE_COUNT:", FACE_COUNT)
        
        bytes_per_v = 4 + 4 + 4 + 1 + 1 + 1
        bytes_per_f = 1 + 4 + 4 + 4 + 4
        v_start = HEADER_SIZE
        v_end = v_start + bytes_per_v * VERTEX_COUNT
        
        v_bytes = data[v_start:v_end]
        f_bytes = data[v_end:]

        for i in range(0, len(v_bytes), bytes_per_v):
            vertex_to_rgb.append(list(struct.unpack("BBB", v_bytes[i+12:i+15]))) # B: uchar
        
        face_id = 0
        for i in range(0, len(f_bytes), bytes_per_f):
            objID, = struct.unpack("<i", f_bytes[i+13:i+17]) # <: little-endian, i: int
            face_to_objID.append(objID)
            
            if objID in objID_to_face:
                objID_to_face[objID].append(face_id)
            else:
                objID_to_face[objID] = [face_id]
                
            face_id += 1
        
    # ASCII PLY
    else:
        print("Parsing ASCII file...")
        data = data.decode("ascii").splitlines()

        # Count number of lines for header (may vary)
        HEADER_SIZE = 1
        VERTEX_COUNT = None
        FACE_COUNT = None
        for line in data:
            if "element vertex" in line:
                VERTEX_COUNT = int(line.split(" ")[-1])
            elif "element face" in line:
                FACE_COUNT = int(line.split(" ")[-1])
            elif "end_header" in line.lower():
                break
                
            HEADER_SIZE += 1

        print("HEADER_SIZE:", HEADER_SIZE)
        print("VERTEX_COUNT:", VERTEX_COUNT)
        print("FACE_COUNT:", FACE_COUNT)
        
        # Extract per vertex color and per face object ID information
        v_start = HEADER_SIZE
        v_end = v_start + VERTEX_COUNT
        vertex_info = data[v_start:v_end]
        face_info = data[v_end:]

        for i, line in enumerate(vertex_info):
            rgb = list(map(lambda x: int(x), line.split(" ")[-3:]))
            vertex_to_rgb.append(rgb)
            
        for i, line in enumerate(face_info):
            objID = int(line.split(" ")[-1])
            face_to_objID.append(objID)
            
            if objID in objID_to_face:
                objID_to_face[objID].append(i)
            else:
                objID_to_face[objID] = [i]
    print("Importing finished.")
        

def mp3d_importer(house_path, ply_path):
    global vertex_to_rgb, face_to_objID, objID_to_face, seg_to_objID
    
    vertex_to_rgb.clear()
    face_to_objID.clear()
    objID_to_face.clear()
    seg_to_objID.clear()

    print("Parsing house file...")

    region_idx = 0
    match = idx_rxp.match(ply_path)
    if match:
      region_idx = int(match.group(1))

    with open(house_path, "r") as file: 
      content = file.read()
      segments = seg_rxp.findall(content)
      for v, k in segments:
        seg_to_objID[int(k)] = int(v)

    # Read binary PLY file
    data = None
    with open(ply_path, "rb") as file:
        data = file.read()

    print("Parsing binary file...")
    
    # Count number of lines for header (may vary)
    HEADER_SIZE = 0
    VERTEX_COUNT = None
    FACE_COUNT = None
    
    line = b""
    start = 0
    max = len(data)
    while line.lower() != "end_header":
        end = data.find(b"\n", start, max)
        line = data[start:end].decode("ascii")
        if "element vertex" in line:
            VERTEX_COUNT = int(line.split(" ")[-1])
        elif "element face" in line:
            FACE_COUNT = int(line.split(" ")[-1])
            
        HEADER_SIZE += end - start + 1
        start = end + 1

    print("HEADER_SIZE (in bytes):", HEADER_SIZE)
    print("VERTEX_COUNT:", VERTEX_COUNT)
    print("FACE_COUNT:", FACE_COUNT)
    
    bytes_per_v = 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 1 + 1 + 1
    bytes_per_f = 1 + 4 + 4 + 4 + 4 + 4 + 4
    v_start = HEADER_SIZE
    v_end = v_start + bytes_per_v * VERTEX_COUNT
    
    v_bytes = data[v_start:v_end]
    f_bytes = data[v_end:]

    for i in range(0, len(v_bytes), bytes_per_v):
        vertex_to_rgb.append(list(struct.unpack("BBB", v_bytes[i+32:i+35]))) # B: uchar
    
    face_id = 0
    for i in range(0, len(f_bytes), bytes_per_f):
        material_id, segment_id, category_id = struct.unpack("<iii", f_bytes[i+13:i+25]) # <: little-endian, i: int

        objID = -1
        if material_id >= 0:
          # Adjust material_id when reading regions
          material_id += region_idx * REGION_OFFSET # Correct location?
          objID = seg_to_objID[material_id]

        face_to_objID.append(objID)
        
        if objID in objID_to_face:
            objID_to_face[objID].append(face_id)
        else:
            objID_to_face[objID] = [face_id]
            
        face_id += 1

def exporter(path):
    global HEADER, mesh, vertex_to_rgb, face_to_objID

    verts = mesh.verts
    faces = mesh.faces
    with open(path, "wb+") as ply:
        header = HEADER.format(vcount=len(verts), fcount=len(faces))
        ply.write(header.encode("ascii"))

        for i, v in enumerate(mesh.verts):
            buf = struct.pack('fffBBB', *v.co, *vertex_to_rgb[i]) # f: float, B: uchar
            ply.write(buf)

        for i, f in enumerate(mesh.faces):
            v_idxs = list(map(lambda x: x.index, f.verts))
            buf = struct.pack('<Biiii', 3, *v_idxs, face_to_objID[i]) # <: little-endian B: uchar, i: int
            ply.write(buf)
            
def get_objID_of_selection():
    """Get object IDs of selected faces"""
    global mesh, face_to_objID

    objIDs = []

    # Face info for selection
    for f in mesh.faces:
        if f.select:
            face_idx = f.index  # Blender idx (zero-indexed) -> PLY file line idx (zero-indexed)
            objID = face_to_objID[face_idx]
            print("Face ID: {} has the object ID: {}".format(face_idx, objID))
            objIDs.append(objID)

    consensus = Counter(objIDs).most_common(1)[0][0]
    return consensus
                
def select_faces_of_obj(objID=None, sel_extend=False):
    """Get all faces bound to a specific object"""
    
    global objID_to_face, mesh, obj_data

    if objID == None:
      objID = get_objID_of_selection()

    # Deselect all previous if extended selection is not desired
    if not sel_extend:
        for f in mesh.faces:
            f.select = False

    # Select faces of a specific object
    idxs = objID_to_face[objID]
    for i in idxs:
        mesh.faces[i].select = True

    bmesh.update_edit_mesh(obj_data, True)
    
def get_objIDs_of_vertex(v):
    """Get all objID's bound to a specific vertex v"""
    
    global face_to_objID

    adj_faces = v.link_faces
    obj_ids = list(map(lambda x: face_to_objID[x.index], adj_faces))
    return obj_ids
    
def cut_object(objID=None):
    """Cuts the faces of objID from its neighbours. Faces whose one or more edges are removed are deleted."""
    
    global objID_to_face, face_to_objID, mesh, obj_data

    if objID == None:
      objID = get_objID_of_selection()

    idxs = objID_to_face[objID]
    del_faces = set()
    del_edges = set()
    # For each face f of object
    for i in idxs:

       # For each vertex v of the face
       for v in mesh.faces[i].verts:

           # For each edge e originating from v
           for e in v.link_edges:
               this = v.index
               v1, v2 = e.verts
               
               # Determine other endpoint of the edge
               that = v1
               if this == that.index:
                   that = v2
               
               # Determine the object IDs bound to the other endpoint
               that_objIDs = get_objIDs_of_vertex(that)
               # Remove edge if other endpoint doesn't belong to same object
               if objID not in that_objIDs:
                   # Required to update face-objID mapping
                   for f in that.link_faces:
                     if e in f.edges:
                        del_faces.add(f.index)

                   del_edges.add(e)

    # Update face-objID mappings
    objID_to_face.clear() # Needs to be rebuilt due to face index shift 
    for face_id in sorted(del_faces, reverse=True):
        del face_to_objID[face_id]

    for face_id, objID in enumerate(face_to_objID):
        if objID in objID_to_face:
            objID_to_face[objID].append(face_id)
        else:
            objID_to_face[objID] = [face_id]


    bmesh.ops.delete(mesh, geom=list(del_edges), context="EDGES_FACES")
    bmesh.update_edit_mesh(obj_data, True)

def get_rotation_matrix(angles, axes, square=True, in_degrees=True):
    r = Rotation.from_euler(axes.upper(), angles, degrees=in_degrees).as_matrix()
    R = np.eye(4)if square else np.eye(3,4)
    R[0:3,0:3] = r
    return R

def get_translation_matrix(t, square=True):
    T = np.eye(4) if square else np.eye(3, 4)
    t = np.array([*t, 1]) if (square and len(t) == 3) else t
    T[:,3] = t
    return T

def unravel(t):
    t = t.reshape(3,4)
    t = np.vstack((t, [0, 0, 0, 1]))
    return t

def get_RT_matrix(angles, t, square=True, in_degrees=True):
    """Specify rotation angles (as (rx, ry, rz)) and translation parameters (as (x,y,z)), return equivalent 3x4 or 4x4 transformation matrix"""
    
    # Get 4x4 rotation matrix: Rx @ Ry @ Rz == 'XYZ'
    RT = get_rotation_matrix(angles, 'XYZ', square=square, in_degrees=in_degrees)
    # Embed translation
    t = np.array([*t, 1]) if (square and len(t) == 3) else t
    RT[:,3] = t
    # Return overall RT matrix as 4x4 or 3x4 depending on square flag
    return RT

def selection_center():
    """Calculate the center of selected faces. Required for pivot point transform during rotation."""
    
    global mesh

    selected_verts = filter(lambda v: v.select, mesh.verts)
    selected_coords = np.array([v.co for v in selected_verts])
    return selected_coords.mean(axis=0)

def selection_center_wrt_habitat(blender_pos=None):
    global rotx90minus, rotx90plus
    
    if not blender_pos:
      blender_pos = selection_center()
    
    converted = rotx90minus @ np.array([*blender_pos, 1])
    
    return converted[0:3]

def translate_selection(t, store=True):
    """Translate with t (given as (x,y,z)) using vertex coordinates of selected faces, return equivalent 4x4 translation matrix"""
    
    global mesh, obj_data, transforms
    
    t = mathutils.Vector(t)
    for v in mesh.verts:
      if v.select:
        v.co += t

    t_matrix = get_translation_matrix(t)
    if store:
        transforms.append(t_matrix)

    bmesh.update_edit_mesh(obj_data, True)
    return t_matrix

def rotate_selection(angles, axes, store=True, in_degrees=True):
    """Rotate vertices with one or more angles around one or more axis ('X','Y','Z'), return equivalent 4x4 rotation matrix."""
    
    global mesh, transforms
    
    pivot = selection_center()
    origin_map = get_translation_matrix(-pivot, square=True)
    R4 = get_rotation_matrix(angles, axes, square=True, in_degrees=in_degrees)
    inv_origin_map = get_translation_matrix(pivot, square=True)
    R4 = inv_origin_map @ R4 @ origin_map

    # Extract 3x4 part and apply rotation on vertices around the center of selection.
    R3 = R4[0:3]
    for v in mesh.verts:
      if v.select:
        v_hom = np.array([*v.co, 1])
        v_hom = (R3 @ v_hom).tolist()
        v.co = mathutils.Vector(v_hom[0:3])

    if store:
        transforms.append(R4)

    bmesh.update_edit_mesh(obj_data, True)
    return R4

def transform_selection(transform, store=True):
    """
    Apply transformation matrix on selected vertices.
    This method can be used for testing the transformation matrix generated with composite transformations.
    Provided transformation matrix can also be stored in transforms if store flag is set.
    """
    global mesh, obj_data, transforms

    if store:
      if transform.shape == (3, 4):
        transform = np.vstack((transform, [0, 0, 0, 1]))
      transforms.append(transform)

    # Extract 3x4 part and apply on vertices
    transform = transform[0:3]
    for v in mesh.verts:
      if v.select:
        v_hom = np.array([*v.co, 1])
        v_hom = (transform @ v_hom).tolist()
        v.co = mathutils.Vector(v_hom[0:3])

    bmesh.update_edit_mesh(obj_data, True)

def compose_transforms(transforms, clear=True):
    """
    Takes a list of 4x4 transformation matrices and returns the overall 4x4 matrix. 
    Assumes that the first item of the list is the first transformation and the last item of the list is the last transformation in the sequence.
    Set clear flag to empty processed list.
    """
    
    composite = np.linalg.multi_dot(transforms[::-1])
    if clear:
        transforms.clear()
    return composite

def inv_transform_selection(transforms):

    global mesh, obj_data

    T = transforms
    if isinstance(transforms, list):
        T = compose_transforms(transforms, clear=False)
    Tinv = np.linalg.inv(T)

    # Extract 3x4 part and apply on vertices
    Tinv = Tinv[0:3]
    for v in mesh.verts:
      if v.select:
        v_hom = np.array([*v.co, 1])
        v_hom = (Tinv @ v_hom).tolist()
        v.co = mathutils.Vector(v_hom[0:3])

    bmesh.update_edit_mesh(obj_data, True)

def convert_to_habitat(T):
    global rotx90minus, rotx90plus
    return rotx90minus @ T @ rotx90plus

def convert_to_blender(T):
    global rotx90plus, rotx90minus
    return rotx90plus @ T @ rotx90minus

def export_moved_info(path, transforms, objID=None, clear=True):
    """Takes a single matrix or a list of matrices, stores and returns overall matrix with respect to habitat-sim convention"""
    
    global SEG_COLORS

    if objID == None:
      objID = get_objID_of_selection()

    transform = transforms
    if isinstance(transforms, list):
        transform = compose_transforms(transforms, clear)
    
    transform = convert_to_habitat(transform)

    info = [{
        "color": SEG_COLORS[objID % 40],
        "name": objID,
        "transformation": transform[:3].ravel().tolist() # Save as 3x4 matrix
    }]
    with open(path, "w+") as file:
        json.dump(info, file, indent=4)
    return transform
