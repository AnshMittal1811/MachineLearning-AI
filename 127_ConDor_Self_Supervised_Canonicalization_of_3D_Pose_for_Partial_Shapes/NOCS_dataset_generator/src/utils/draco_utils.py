import numpy as np
from scipy.spatial.transform import Rotation as rot_utils
import json

# File borrowed and edited from https://github.com/RahulSajnani/DRACO-Weakly-Supervised-Dense-Reconstruction-And-Canonicalization-of-Objects

def construct_camera_matrix(focal_x, focal_y, c_x, c_y):
    '''
    Obtain camera intrinsic matrix
    '''

    K = np.array([[focal_x,       0,     c_x],
                  [0, focal_y,     c_y],
                  [0,       0,       1]])

    return K


def get_grid(x, y):
    '''
    Get index grid from image
    '''

    y_i, x_i = np.indices((x, y))
    coords = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(x*y, 3)

    return coords.T

def extract_depth_pointcloud(depth, mask = None):
    '''
    Takes a depth image and extracts its point cloud
    '''

    # Using camera intrinsics from DRACO
    K = construct_camera_matrix(888.88, 1000, 320, 240)
    invK = np.linalg.inv(K)

    return depth_2_point_cloud(invK, depth, mask)


def depth_2_point_cloud(invK, depth_image, image_mask):
    '''
    Convert depth map to point cloud
    '''
    # print(depth_image.shape)
    points_hom = get_grid(depth_image.shape[0], depth_image.shape[1])

    depth = depth_image.reshape(1, -1)
    if image_mask is not None:
        depth_mask = image_mask.reshape(1, -1) >= 0.5
        # print("using nocs mask")
    else:
        depth_mask = depth <= 20
    point_3D = invK @ points_hom
    point_3D = point_3D / point_3D[2, :]
    point_3D = point_3D[:, depth_mask[0]] * depth[:, depth_mask[0]]
    # point_3D = point_3D[:, depth_mask[0]]
    return point_3D.T


def read_json_file(path):
    '''
    Read json file
    '''
    json_data = []

    with open(path) as fp:
        for json_object in fp:
            json_data.append(json.loads(json_object))

    return json_data

def pose_to_mat(json_data):

    # print(json_data)
    pose = np.array([
                    json_data[0]['position']['x'],
                    json_data[0]['position']['y'],
                    json_data[0]['position']['z'],
                    json_data[0]['rotation']['x'],
                    json_data[0]['rotation']['y'],
                    json_data[0]['rotation']['z'],
                    json_data[0]['rotation']['w']
                     ])



    # print(pose)
    rot_mat = rot_utils.from_quat(pose[3:]).as_matrix()  # num_views 3 3
    translation_mat = np.expand_dims(pose[:3], axis = 1) # num_views 3 1
    # print(rot_mat.shape, translation_mat.shape)

    transformation_mat = np.concatenate([rot_mat, translation_mat], 1)
    transformation_mat = np.concatenate([transformation_mat, np.array([[0,0,0,1]])], 0)

    
    return transformation_mat


def solve_orthogonal_procrustes(pts_1, pts_2):
    '''
        Peform one iteration of closest point to obtain the transformation matrix

        pts_1, pts_2 dims = 3 x n_points
        R @ pts_2 + t = pts_1
        '''

    dims, n_points = pts_1.shape
    center_1 = np.mean(pts_1, axis=1)[..., None]
    center_2 = np.mean(pts_2, axis=1)[..., None]

    # Centering the points
    pts_1_centered = pts_1 - center_1
    pts_2_centered = pts_2 - center_2

    # Performing SVD to solve for rotation
    H = pts_1_centered @ pts_2_centered.T
    U, S, Vt = np.linalg.svd(H)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        # Reflection case
        Vt[dims - 1, :] *= -1
        R = U @ Vt

    # Computing translation
    t = center_1 - R @ center_2

    T = np.eye(R.shape[0] + 1)
    T[:dims, :dims] = R
    T[:dims, dims] = t[:, 0]

    return T, R, t
