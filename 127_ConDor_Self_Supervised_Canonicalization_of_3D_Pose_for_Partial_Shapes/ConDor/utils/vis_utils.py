import seaborn as sns
import open3d as o3d
import numpy as np
import tensorflow as tf
import argparse, os, sys
import glob, os

def create_color_samples(N):
    '''
    Creates N distinct colors
    N x 3 output
    '''

    palette = sns.color_palette(None, N)
    palette = np.array(palette)

    return palette

def convert_tensor_2_numpy(x):
    '''
    Convert pytorch tensor to numpy
    '''
    
    out = tf.squeeze(x, axis = 0).numpy()
    
    return out 


def save_pointcloud(x, filename = "./pointcloud.ply"):
    '''
    Save point cloud to the destination given in the filename
    x can be list of inputs (Nx3) capsules or numpy array of N x 3
    '''

    label_map = []
    if isinstance(x, list):
        
        pointcloud = []
        labels = create_color_samples(len(x))
        for i in range(len(x)):
            pts = x[i]
            # print(pts.shape, "vis")
            pts = convert_tensor_2_numpy(pts)

            pointcloud.append(pts)
            label_map.append(np.repeat(labels[i:(i + 1)], pts.shape[0], axis = 0))
        
        # x = np.concatenate(x, axis = 0)
        pointcloud = np.concatenate(pointcloud, axis = 0)
        x = pointcloud.copy()
        label_map = np.concatenate(label_map, axis = 0)
    else:
        x = convert_tensor_2_numpy(x)
        # print(x.shape)
        label_map = np.ones((len(x), 3)) * 0.5

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    pcd.colors = o3d.utility.Vector3dVector(label_map)

    o3d.io.write_point_cloud(filename, pcd)


def distribute_capsules(inv, caps):
    '''
    Distribute pointcloud into different capsules
    '''
    num_caps = caps.shape[-1]

    idx = tf.argmax(caps, axis = -1)   
    idx = tf.squeeze(idx, axis = 0) # N vec
    out = []
    for i in range(num_caps):
        boolean_selection_idx = (idx == i)
        pcd = tf.boolean_mask(inv, boolean_selection_idx, axis = 1)
        out.append(pcd)

    return out


def visualize_outputs(base_path, pointcloud_name, start = 0, max_num = 10, spacing = 1.0, skip = 1):
    '''
    Visualize point clouds in open3D 
    '''

    filename = glob.glob(os.path.join(base_path, "") + "**_" + pointcloud_name)
    filename.sort()
    filename_2 = []
    for f in filename:
        # print(os.path.join(base_path, "") + f.split("/")[-1].split("_")[0] +"_" + pointcloud_name, f)
        if (os.path.join(base_path, "") + f.split("/")[-1].split("_")[0] +"_" + pointcloud_name) == f:
            filename_2.append(f)
    filename = filename_2
    filename = filename[start::skip]
    filename = filename[:max_num]
    # print(filename)
    num_pcds = len(filename)
    if skip != 1:
        num_pcds = len(filename) // skip
    rows = np.floor(np.sqrt(num_pcds))
    pcd_list = []
    pcd_iter = 0
    pcd_index = 0
    for pcd_file in filename:
        if pcd_iter % skip == 0:
        
            pcd = o3d.io.read_point_cloud(pcd_file)
            column_num = pcd_index // rows
            row_num = pcd_index % rows
            vector = (row_num * spacing, column_num * spacing, 0)
            # print(vector)
            pcd.translate(vector)
            pcd_list.append(pcd)
            pcd_index += 1

        pcd_iter +=1
    
    o3d.visualization.draw_geometries(pcd_list)

if __name__ == "__main__":
    

    # Argument parser
    parser = argparse.ArgumentParser(description = "Visualization script")
    parser.add_argument("--base_path", help = "Base path to folder", required = True)
    parser.add_argument("--pcd", help = "PCD string to visualize", required = True)
    parser.add_argument("--spacing", help = "Spacing", default = 2.0, type = float)
    parser.add_argument("--num_list", help = "indices", nargs="+", default = list(range(9)), type = int)
    parser.add_argument("--start", help = "start index", default = 0, type = int)
    parser.add_argument("--num", help = "number of models", default = 9, type = int)
    parser.add_argument("--skip", help = "number of models to skip", default = 1, type = int)


    args = parser.parse_args()
    #######################################################################
    

    visualize_outputs(args.base_path, args.pcd, spacing = args.spacing, start = args.start, max_num = args.num, skip = args.skip)
